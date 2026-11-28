# Design: media ingestion v2, covering video scenes and music details

**Status:** implemented and tested: v2a (scenes), v2b (music) and v2c (CLAP tags, `--tags`).
Per-section music documents are not implemented. Builds on v1 ([MEDIA_INGESTION_DESIGN.md](MEDIA_INGESTION_DESIGN.md)).
How to run: [RUNNING.md §7](RUNNING.md#7-audio-and-video-datasets). **Branch:** not needed: no
shared code changed, and the extras only add packages (§7).

**Differences from this design, found while building it:**
* **Scene cuts compare colour, not grey.** Demo scenes of similar brightness but different
  colour were missed in grey (difference 0.06 against a 0.12 threshold).
* **The captioner is given on the command line** (`--captioner`, `--captioner-url`) with its own
  small client, not as a `models.captioner` config role. No existing config or config hash
  changes, and the shared provider code, which counts tokens on string messages only, stays
  untouched. Like RAGAS, it accepts local, unpriced servers only.
* **Laptop captioner:** IBM Granite Vision 3.2 2B (`granite3.2-vision:2b`, Apache-2.0, 2.4 GB),
  built for reading text in images. Qwen2.5-VL-7B stays the GPU choice.
* **Key confidence:** a key is reported only at confidence ≥ 0.5. Drums-only tracks (about 0.3)
  are reported as "key unclear".
* **Music under speech:** a quiet beat under louder speech scored 0.27 against the 0.3 threshold
  and is reported as speech. That is conservative by design, and the threshold was not tuned to
  this one example.
* **Music details live on the clip card** (text + metadata). Separate per-section
  `kind: music` documents are not implemented.
* **OpenMP:** a real check finds *no* OpenMP runtime loaded by the media + music path (PyTorch,
  as a positive control, loads `libomp.dylib` and is detected).
* **v2c tags use `laion/clap-htsat-fused`**, the CLAP release with safetensors weights. The
  others ship only pickle `.bin` files, which are unsafe to load. Tags are computed in a
  subprocess, so PyTorch never meets CTranslate2, and the audio is decoded with PyAV directly
  (importing faster-whisper would load CTranslate2).
* **Tags are gated by the librosa content label, not by CLAP.** A closed label list always picks
  a genre (a talk came out "jazz, 0.88"). CLAP's own music/speech scores were wrong on the demo:
  the instrumental was rated "noise" 0.59, and a talk "music" 0.53. So `--tags` requires
  `--music`, and only clips labelled music or mixed are tagged.

## 1. Goal

v1 makes what is **said** in a clip searchable. v2 adds what is **shown** (video scenes) and what
the **music** is like (tempo, key, loudness, and whether a clip is speech, music or mixed). Each
new stage only adds documents to the clip, with a new `metadata.kind`. Retrieval, both judges,
RAGAS and the run records stay unchanged, and answers still cite clip and time.

```
clip ──► v1: probe + transcript windows + clip card
     ├─► v2a: scene detection ──► one keyframe per scene ──► caption (vision-language model)
     │        └─► kind "scene" documents: "talk.mp4 [01:12–01:40] Scene: a woman points at a telescope …"
     └─► v2b: music analysis ──► kind "music" document + facts added to the clip card
              "track.mp3 is music, about 128 BPM, key A minor, mean level -18 dBFS, 3 min 12 s"
```

Questions this enables: "What is shown when the dome opens?", "Which clip shows a mirror being
polished?", "Which track is the fastest?", "Which clips contain music?". The last two are
comparisons across clip cards, which should favour graph and global modes.

## 2. v2a: video scene captions

| Part | Choice | Why |
|---|---|---|
| Scene detection | **Our own**, on frames decoded with PyAV (already installed): downscale to 64×36 grey, cut where the mean absolute frame difference exceeds a threshold (default 0.12), at least `--min-scene-s` (default 2 s) per scene, and at most `--max-scene-s` (default 30 s; longer scenes are split) | no new dependency. PySceneDetect 0.7.3 (BSD-3) is the fallback if this proves too crude, with OpenCV as an extra dependency |
| Keyframe | the frame nearest the scene's middle, saved as JPEG (Pillow, already a dependency) under the cache | the middle frame avoids transition blur |
| Captioner | **Qwen2.5-VL-7B-Instruct** (Apache-2.0, ungated; HF `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f…`, full SHA pinned at implementation), served like the other models: **vLLM on GPU** or **Ollama `qwen2.5vl:7b`** on a laptop (both verified to exist 2026-11-27) | an OpenAI-compatible endpoint, so no new provider code, no PyTorch in our process, and calls are cached and budgeted like every LLM call |
| Not chosen | Qwen2.5-VL-3B (its card states no Apache licence, unlike the 7B), SmolVLM-256M/500M (Apache-2.0, very small: a fallback for CPU-only machines), Florence-2 (MIT, needs `trust_remote_code`) | |
| Prompt | "Describe what is visible in this video frame in 1–3 factual sentences: people, objects, actions, any readable text. Do not guess names." Temperature 0, seed 0 | short, factual, no invented identities |

Each scene becomes one document:

```json
{"doc_id": "tour-2bd2223c#s000012-000025", "title": "telescope_tour.mp4 [00:12–00:25] (scene)",
 "text": "[00:12–00:25] Scene: a large circular mirror on a workbench; a person in a lab coat polishes it.",
 "metadata": {"kind": "scene", "clip_id": "tour-2bd2223c", "start_s": 12.0, "end_s": 25.0,
              "keyframe_s": 18.5, "captioner": "Qwen/Qwen2.5-VL-7B-Instruct@cc594898"}}
```

* **Images go to the captioner as base64 `image_url` content**, the OpenAI vision message format
  that vLLM and Ollama both accept. This must be verified against both servers first; it is the
  main technical risk of v2a.
* **Caching:** keyframes and captions are cached by (file SHA-256, scene bounds, model,
  revision, prompt version), so changing the window or scene settings re-captions only the
  scenes that changed.
* **Gold answers:** questions can name a time span exactly as in v1. Scene documents
  overlapping the span become gold, alongside transcript windows (`"kinds": ["scene"]` restricts
  it).
* **Model independence:** the captioner is Qwen, like the generator. That is fine for
  preprocessing, because the judges (phi-4 / Llama, Granite) stay from other families; the
  manifest records the captioner.
* **Cost and time:** one caption per scene. A 10-minute video with a cut every ~10 s is ~60
  calls. Measure on 2–3 clips before a big folder (as with `index pilot`).

## 3. v2b: music details

| Feature | How (librosa 0.11.0, ISC) | Stored as |
|---|---|---|
| speech / music / mixed | share of the clip that Whisper's voice-activity filter kept as speech (v1 already logs it) combined with librosa's harmonic–percussive energy ratio | `content: speech \| music \| mixed \| silence` |
| tempo | `librosa.beat.beat_track` on the onset envelope → BPM, plus beat count | `tempo_bpm` (music/mixed only) |
| key | chroma (`chroma_cqt`) averaged over the clip, correlated with the Krumhansl–Schmuckler major/minor profiles → e.g. "A minor", with the correlation as confidence | `key`, `key_confidence` |
| loudness | RMS in dBFS (mean and peak); integrated LUFS would need `pyloudnorm`, so it is left out | `rms_dbfs_mean`, `peak_dbfs` |
| brightness | mean spectral centroid in Hz | `spectral_centroid_hz` |

* **Clip card:** the card gains a sentence ("Music: about 128 BPM, key A minor (confidence
  0.71), mean level -18 dBFS."). A separate `kind: "music"` document carries the same facts per
  60 s section for long tracks, so "where does the tempo change?" is answerable.
* **Lyrics** are already transcribed by v1. Whisper's music handling (VAD, the hallucination
  filter) keeps false lyrics down; the manifest reports the speech share so music-heavy clips
  can be checked.
* **Optional, later (v2c): genre, mood and instrument tags** by zero-shot audio–text matching
  with CLAP (`laion/clap-htsat-unfused`, Apache-2.0). It needs PyTorch, so it must run as its
  own command and process (`data media-tags`), never inside `ingest-media` (§6).

## 4. Command

```bash
uv sync --locked --extra media --extra music
uv run elevate-rag data ingest-media clips/ --as my_media \
    --scenes --captioner-config configs/pipelines/local_7b.yaml   # models.captioner in that config
    --music                                                        # tempo, key, loudness, content
```

* `--scenes` and `--music` are off by default, so v1 behaviour and dataset versions are
  unchanged unless asked for.
* The captioner is a new optional model role, `models.captioner`, in the pipeline configs. Like
  `judge2`, it is dropped from the config hash when unset, so existing run ids do not change.
  It only affects the *dataset* (whose version changes when captions are added), not the
  pipeline.

## 5. Evaluation

| What | How |
|---|---|
| answers on scene/music questions | as v1: EM/F1, two judges, RAGAS, run records; the question set gains `query_type` `comparison` / `global` items across clip cards |
| tempo and key accuracy | unit tests on generated audio: a click track at 90 / 120 / 150 BPM (pass if within ±2 BPM), and a sustained A-minor triad / C-major scale (key correct). An integration test checks the same on PyAV-decoded MP3s |
| scene detection | generated video with known cuts (v1's demo already makes colour changes per sentence: pass if every cut is found within ±0.5 s, with no extras) |
| caption faithfulness | no automatic ground truth. The demo video gets frames with drawn text ("MIRROR", "DOME"), so a caption mentioning the right text is a checkable proxy. The report states that caption quality is unmeasured beyond that |

## 6. Process isolation and dependencies

* **Libraries:** librosa 0.11.0 needs numba / llvmlite.
  * Numba has an OpenMP threading layer that could load a second OpenMP runtime next to
    CTranslate2's. The macOS OpenMP abort that `utils.guard_openmp` prevents for FAISS + PyTorch
    is exactly this risk.
  * Mitigation: `ingest-media` sets `NUMBA_THREADING_LAYER=workqueue` before importing librosa,
    and an isolation test asserts that no `libomp` / `libiomp` is loaded twice (the same style as
    v1's "no torch/faiss" test). If that cannot be guaranteed, music analysis runs in a
    subprocess.
* **Lock check (2026-11-27, on a copy):** adding `music = ["librosa>=0.11,<1"]` only adds 15
  packages (librosa 0.11.0, numba 0.68.1, llvmlite 0.50.0, soundfile 0.14.0, soxr 1.1.0,
  audioread, pooch, msgpack, …). No existing version changes.
* librosa 1.0 needs Python 3.12+, while the project supports 3.11, so 0.11 it is.
* **v2a adds no Python dependency:** scene detection uses PyAV + numpy, keyframes use Pillow,
  and captions go through the existing OpenAI-compatible client.

## 7. What must not break

The same rules as v1 (design §9), plus:
* **Flags off by default:** `--scenes` and `--music` are off, and v1 datasets and their versions
  stay identical (test: re-ingest the demo set without flags and get the same dataset version).
* **`models.captioner` unset** leaves every config hash unchanged (test, as for `judge2`).
* **Separate extras:** the `music` extra is separate from `media`, so speech-only users don't
  install numba.

## 8. Implementation plan (about 2–3 days, M3 + M4)

| Step | Who | Done when |
|---|---|---|
| 1. Verify vision messages with vLLM (Qwen2.5-VL-7B) and Ollama (`qwen2.5vl:7b`); pin the full revision | M3 | one keyframe captioned through `OpenAICompatLLM` on both |
| 2. Scene detection + keyframes (PyAV, numpy) with the generated-video test | M4 | cuts found within ±0.5 s |
| 3. `models.captioner` role, caption cache, `kind: scene` documents, gold mapping for scenes | M3 | unit tests with a fake captioner; config hashes unchanged |
| 4. Music features + card sentence + `kind: music` documents; OpenMP isolation test | M4 | the BPM / key tests pass; isolation test passes |
| 5. Demo: `make_media_demo.py` adds drawn text to video frames, a click-track clip and a short melody; questions on scenes and music | M4 | demo ingests with `--scenes --music` |
| 6. Experiment on the demo set (V4, local, global, mix), docs, PROGRESS | M3 + M5 | run records; RUNNING §7 updated |

## 9. Open questions

1. Is the frame-difference threshold good enough on real footage (fades, camera motion)? If
   not, switch to PySceneDetect's adaptive detector.
2. Caption length and prompt: 1–3 sentences keeps scene documents small. Check that the
   captioner does not invent names (the prompt forbids it; spot-check the demo).
3. Should scene captions and the transcript of the same span be merged into one document? That
   gives one chunk per span (simpler citations), but mixes two sources. Start separate, and
   compare on the dev split.
