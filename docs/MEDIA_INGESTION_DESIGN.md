# Design: speech-to-text ingestion for audio and video (v1)

**Status:** steps 1–3, 5 and 6 of [§10](#10-implementation-plan-about-2-days-for-m3-with-m4-in-parallel)
are implemented and tested. Step 4, the team's own clips and questions, is open; a synthetic demo
set exists (`scripts/make_media_demo.py`). Run instructions are in
[RUNNING.md §7](RUNNING.md#7-audio-and-video-datasets). **Owner:** media lead (M3), with M4 for
the clip and question set. **Branch:** not needed (see [§9](#9-how-existing-things-stay-unbroken)).

**Differences from the design, found while building it:**
* PyAV is pinned below 19: PyAV 19 removed an argument that faster-whisper 1.2.1 passes.
* A window's overlap segment is repeated only when it is shorter than half a window. Otherwise a
  long segment would fill a whole extra window.
* The device class and compute type (int8 on CPU, float16 on GPU) are part of the transcript
  cache key and the manifest, because they change transcripts slightly.
* `.aiff` audio is also accepted.
* **Added: a vocabulary hint (`vocabulary.txt`, `--vocabulary`).** It is passed as
  faster-whisper's `hotwords` and stored in the cache key and manifest. Without it, both `small`
  and `large-v3` misspelt every invented name in the demo. With it, `small` spelt all six right
  (word error rate 6–15 % → 2–10 %).

## 1. Goal

Turn a folder of audio and video clips into a normal Elevate-RAG dataset. Every existing part
then works on it unchanged:
* indexing, both vector and graph;
* every retrieval mode and router;
* official EM/F1, the two judges and RAGAS;
* run records.

Answers cite the clip and the time ("`interview.mp4` 03:00–04:00").

```
clips/ ──► probe (PyAV) ──► decode 16 kHz mono ──► transcribe (faster-whisper, pinned model)
   │                                                        │ timestamped segments
   │                                                        ▼
   └─► licences.csv                                   time windows (~60 s)
                                                            │
questions.jsonl ──► gold windows from time spans            ▼
                         └──────────► write_processed ──► data/processed/<NAME>/
                                                            (corpus, questions, splits, manifest)
```

**In scope (v1):**
* speech in any language Whisper supports;
* technical details of each clip: duration, container, codecs, sample rate, channels,
  resolution, frame rate, and container tags such as title or artist;
* a question set with time-span gold answers;
* a transcript cache;
* tests.

**Out of scope (later versions, designed for, see [§11](#11-extension-points)):**
* video scene captions;
* music features (tempo, key);
* speaker diarisation;
* on-screen text.

## 2. Command

```bash
uv sync --locked --extra media            # plus your usual extras
uv run elevate-rag data ingest-media clips/ --as media_demo \
    --questions clips/questions.jsonl \
    --model small                           # small (laptop, default) | large-v3 (GPU)
    [--language en] [--window-s 60] [--device auto] [--licences clips/licences.csv]
```

* **Output:** `data/processed/media_demo/`, the same layout and manifest as every other dataset.
  From there the usual commands just work: `index build PIPELINE --set dataset=media_demo`,
  `ask`, `experiment run … --datasets media_demo`.
* **Re-running is cheap.** Transcripts are cached per (file SHA-256, model, revision,
  parameters), so a second run only re-windows and re-writes. Changing `--window-s` needs no
  re-transcription.
* **Media extensions:** audio `.wav .mp3 .m4a .flac .ogg .opus`; video `.mp4 .mov .mkv .webm`.
  Other files are listed in the manifest as skipped, never silently dropped.
* **Licences are required.** Every clip needs a row in `licences.csv`
  (`file,licence,source_url,notes`), and clips without one are refused. Licences go into the
  manifest.

## 3. Models (verified 2026-11-27)

| Name | Hugging Face repo @ revision | Size | Licence | Use |
|---|---|---|---|---|
| `small` (default) | `Systran/faster-whisper-small` @ `536b0662742c02347bc0e980a01041f333bce120` | 0.49 GB | MIT, ungated | laptops (CPU, int8), tests |
| `large-v3` | `Systran/faster-whisper-large-v3` @ `edaa852ec7e145841d8ffdb056a99866b5f0a478` | 3.09 GB | MIT, ungated | GPU (CUDA, float16), the reported results |

These are CTranslate2 conversions of OpenAI's Whisper models, used through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper):
* **Libraries:** faster-whisper 1.2.1 (MIT, 2025-10-31), CTranslate2 4.9.0 and PyAV 18.0.0
  (BSD-3; 19 is incompatible with faster-whisper 1.2.1). PyAV ships its own FFmpeg libraries, so **no system ffmpeg is needed**; this Mac has
  none.
* **Model download:** `faster_whisper.download_model(repo, revision=…, cache_dir=$HF_HOME)`
  pins the revision.
* **The existing `scripts/download_models.py` skips `.bin` files** to save space, so it must not
  be reused for these models: their weights are `model.bin`.
* **Hardware:** on macOS CTranslate2 runs on CPU only (no Metal). Measure speed on 2–3 clips
  before ingesting many.
* **Alternative considered:** NVIDIA Parakeet-TDT-0.6B-v2. It's CC-BY-4.0, English-only, and
  needs NeMo, a large dependency. Rejected for v1.

## 4. Transcription settings

Settings are chosen for reproducible, faithful transcripts. All are recorded in the manifest.

| Setting | Value | Why |
|---|---|---|
| `temperature` | `0.0` (no fallback list) | deterministic decoding |
| `beam_size` | 5 | faster-whisper default; stable |
| `vad_filter` | `True` (Silero VAD, bundled) | skips silence and music-only stretches, where Whisper tends to hallucinate |
| `condition_on_previous_text` | `False` | stops one error from repeating through a long clip (a known Whisper loop) |
| `word_timestamps` | `False` | segment times (a few seconds each) are enough for 60 s windows |
| `language` | auto-detect unless `--language` | the detected language and its probability are stored per clip |
| `hotwords` | the terms in `vocabulary.txt`, if present | names the model has never heard (people, places, products) are otherwise misspelt |
| `compute_type` | `int8` on CPU, `float16` on CUDA | standard faster-whisper choices |

Each segment kept has: `start`, `end`, `text`, `avg_logprob` and `no_speech_prob`. Segments with
`no_speech_prob > 0.6` and `avg_logprob < -1.0` are dropped as likely hallucinations (Whisper's
own thresholds), and the number dropped is reported.

## 5. Documents

There are two kinds of document per clip, both in the existing `Document` schema
(`types.py`; no schema change).

**Transcript windows.** Consecutive segments are grouped into windows of about `--window-s`
(default 60 s, hard maximum 90 s):
* windows are cut only at segment boundaries;
* the last segment of a window is repeated at the start of the next, so an answer spanning a cut
  is still retrievable.

At roughly 150–180 spoken words a minute, a window is about 250 tokens, below the 512-token
chunk size. The chunker therefore keeps **one chunk per window**, and every retrieved chunk maps
to exactly one time span.

```json
{"doc_id": "interview-3f9a1c2e#t000060-000120",
 "title": "interview.mp4 [01:00–02:00]",
 "text": "[01:00] So when we started the lab in 1998 ... [01:52] ... and that was the turning point.",
 "source": "media",
 "metadata": {"kind": "transcript", "clip_id": "interview-3f9a1c2e", "file": "interview.mp4",
              "start_s": 60.0, "end_s": 120.0, "language": "en", "segments": 14,
              "asr_model": "Systran/faster-whisper-small@536b0662"}}
```

**Clip card**, one per clip. Its text is the technical details as sentences, so they can be
retrieved and asked about ("How long is the interview?").

```json
{"doc_id": "interview-3f9a1c2e#card", "title": "interview.mp4 (clip details)",
 "text": "interview.mp4 is a video clip, 12 min 31 s long, MP4 container, H.264 video 1920x1080 at 25 fps, AAC audio 48 kHz stereo. Spoken language: English. Title tag: Lab history interview.",
 "source": "media", "metadata": {"kind": "card", "clip_id": "interview-3f9a1c2e", "duration_s": 751.2, "...": "..."}}
```

* **`clip_id`:** the filename slug plus the first 8 hex digits of the file's SHA-256. It's
  stable, unique, and changes if the file changes.
* **Inline time stamps** (`[01:00]`) at segment starts keep the time visible to the generator,
  which cites passages. The window time also sits in the title, which the chunker puts at the
  start of each chunk.
* **No `::` inside a `doc_id`**, because chunk IDs are `doc_id::c0000`.

## 6. Questions

`questions.jsonl` is written by the team (M4), one object per line:

```json
{"question": "In which year was the lab founded?", "answers": ["1998"],
 "clip": "interview.mp4", "start_s": 62, "end_s": 70, "query_type": "single_hop",
 "evidence": "when we started the lab in 1998"}
```

* **Gold windows:** `gold_doc_ids` = every transcript window overlapping
  `[start_s, end_s]`, or the clip card for questions about technical details (`"card": true`).
* **Gold evidence:** `gold_evidence` is the optional `evidence` quote, used for evidence recall.
  A quote missing from the transcript is reported as a warning, which surfaces ASR errors.
* **Unanswerable questions:** `"answerable": false` with no clip, to test refusals.
* **Validation:** questions are validated and split 20 % dev / 80 % test by the existing
  `write_processed`. A question naming an unknown clip, or a time outside it, is rejected and
  reported.
* **Target size:** 20–30 questions on 10–20 clips, mixed by type:
  * single fact;
  * across two clips (multi-hop);
  * clip details;
  * unanswerable.

## 7. Evaluation

* **Answers:** scored exactly like every other dataset: EM/F1 (primary), the two judges with
  validation status, RAGAS and a run record. No new metric code is needed.
* **Transcription quality (optional v1.1):** if `references/<file>.txt` holds a human-checked
  transcript for a clip, the manifest reports its word error rate (WER). A word-level edit
  distance takes about 20 lines, so no new dependency is needed.
* **External WER check (later):** measure the pinned model's WER on a small sample of a public
  ASR test set (LibriSpeech test-clean, CC-BY-4.0). This mirrors the GroUSE check on the judges.

## 8. Code layout

| File | Contents | Imports heavy libraries? |
|---|---|---|
| `src/elevate_rag/data/media.py` (new) | discovery, licences, windowing, document and question building, the `ingest_media()` orchestration | **no**: pure Python, testable in CI |
| `src/elevate_rag/data/media_backends.py` (new) | `probe(path)` (PyAV) and `WhisperTranscriber` (faster-whisper), both behind a small `Transcriber` protocol | yes, imported lazily only by `ingest-media` |
| `src/elevate_rag/cli/main.py` | one new command, `data ingest-media` | no |
| `pyproject.toml` | new extra `media = ["faster-whisper>=1.2.1,<2"]` | – |
| `tests/unit/test_media.py` (new) | windowing, overlap, IDs, gold mapping, licence refusal, cache, manifest, using a fake transcriber | no |
| `tests/integration/test_media_backends.py` (new) | probe of a generated WAV (stdlib `wave` + numpy); real `small` model on a short public-domain clip downloaded with a pinned SHA-256 | yes; marked `integration` + `network`, skipped without the extra |
| `docs/RUNNING.md` | new section "Audio and video datasets" | – |

**macOS process isolation.** `ingest-media` must never import PyTorch or FAISS. CTranslate2 and
ONNX Runtime bring their own OpenMP runtime, and the macOS FAISS/PyTorch OpenMP clash
(`utils.guard_openmp`) must not get a third member. Ingestion and indexing are separate commands,
and so separate processes, already. A unit test asserts that importing `data/media.py` loads
neither library.

## 9. How existing things stay unbroken

| Risk | How it is avoided | How it is checked |
|---|---|---|
| New dependencies change existing package versions | an optional extra only | **checked 2026-11-27** on a copy of the lockfile: adding the extra only *adds* `faster-whisper 1.2.1`, `ctranslate2 4.9.0`, `av 19.0.2` (18.0.0 for Python < 3.12), `onnxruntime 1.31.1`, `flatbuffers`; no existing version changes |
| Existing datasets or run ids change | no change to `Document`, `Question`, the chunker, `write_processed`, any pipeline config or the config hash; ingestion parameters live in the *dataset* manifest | the full suite (140 tests) passes; dry-run run ids of G1 / G1-7B unchanged |
| CI breaks or slows | CI installs `--extra faiss` only; the media core is pure Python and tested there; backend tests skip without the extra | CI green |
| Large or copyrighted files reach git | clips stay outside the repo or under `data/` (git-ignored); tests generate audio or download a public-domain clip at test time | `git status` clean of media files |
| OpenMP clash on macOS | the separate-process rule in §8 | import test |

**Branch decision:** a separate branch (e.g. `nova-media`) is **not needed**. The work is
additive and the lockfile change is additions only. Create one only if the implementation needs
any of these:
* a change to shared code beyond registering the CLI command, such as the chunker,
  `write_processed` or the schemas;
* an existing package version to move;
* the full test suite to fail at any point.

## 10. Implementation plan (about 2 days for M3, with M4 in parallel)

| Step | Who | Done when |
|---|---|---|
| 1. `media.py`: windowing, IDs, cards, gold mapping, licences, with a fake transcriber | M3 | unit tests pass in CI without the extra |
| 2. `media_backends.py`: probe + Whisper transcriber + transcript cache | M3 | the integration test transcribes the public-domain clip; the second run hits the cache |
| 3. `data ingest-media` CLI, manifest fields, skipped-file report | M3 | ingest of 2–3 real clips → `index build` → `ask` cites clip + time |
| 4. clips (owned or openly licensed), `licences.csv`, `questions.jsonl` | M4 | 20–30 validated questions |
| 5. experiment: `G1_smoke_local_7b`-style spec on the media dataset (V4, naive, local, mix) | M3 + M5 | run records written; EM/F1, judges, RAGAS reported |
| 6. RUNNING.md section, README link, PROGRESS entry | M3 | docs reviewed |

## 11. Extension points

The next two stages (video scene captions, music details) are designed in
[MEDIA_V2_DESIGN.md](MEDIA_V2_DESIGN.md).

Later stages each add more documents per clip, with a new `metadata.kind`. Nothing downstream
changes:

| Stage | `kind` | Likely tools (to be verified then) |
|---|---|---|
| video scene captions | `scene` | scene detection + an open vision-language model describing one keyframe per scene, with time spans |
| music details | `music` | tempo, key, loudness, tags, written into the clip card |
| speaker turns | `transcript` + `speaker` field | an open diarisation model |
| on-screen text | `ocr` | OCR on keyframes |

Linking people and places across clips is where graph retrieval should help most. That comes
for free: LightRAG extracts entities from the transcript windows like any other text.

## 12. Open questions

1. **Window length:** 60 s is a guess. Try 30 / 60 / 90 s on the question set and keep the best
   dev-split result. This is an ablation, so it is decided on dev only.
2. **Hallucination thresholds** (§4): check on the clips, especially music-heavy ones.
3. **Model for the reported numbers:** use `large-v3` on the EC2 GPU. A transcript cache built
   on the laptop with `small` is a different dataset version and must not be mixed with it.
