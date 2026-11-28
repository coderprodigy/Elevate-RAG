# Audio/video ingestion on the synthetic demo set (2026-11-27)

**Verdict:**
* **Pipeline:** the media pipeline works end to end. Clips (WAV, MP3, MP4 video) are
  transcribed, indexed (vectors and knowledge graph), retrieved, answered with clip and time
  citations, and scored by the same evaluator as every other dataset.
* **Main finding:** transcription of names, not retrieval, decides answer quality. A small name
  vocabulary raised exact match from **0.36 to 0.82**, and a five-times-larger speech model did
  not help.
* **Not research results:** these are synthetic clips with 12 questions and an unvalidated judge.

## Setup

| | |
|---|---|
| Clips | 3 synthetic clips from `scripts/make_media_demo.py` (macOS `say`, fictional facts): `observatory_history.wav` 35 s, `river_festival.mp3` 17 s, `telescope_tour.mp4` 25 s (H.264 + AAC) |
| Questions | 12: 7 single-fact, 2 cross-clip multi-hop, 1 comparison, 1 clip-detail (from the clip card), 1 unanswerable; gold time spans exact by construction |
| Transcription | faster-whisper 1.2.1, `Systran/faster-whisper-small` (pinned), CPU int8, 15 s windows (max 25 s) |
| Answering | `local_7b.yaml`: Qwen2.5-7B generator/extractor, bge embedder + reranker, Llama-3.1-8B judge (one judge: the second judge, Granite, was not pulled to save disk) |
| Spec | `configs/experiments/MEDIA_demo_local_7b.yaml` (variants V4, naive, local, mix; exact vectors + NetworkX) |
| Cost | $0; laptop (16 GB Apple silicon) |

## Results (11 answerable questions + 1 unanswerable; one question ≈ 9 points)

| Transcripts | Variant | EM | F1 | Judge accuracy¹ | Recall@10 |
|---|---|---|---|---|---|
| `small`, no vocabulary | V4 / naive / mix | 0.36 | 0.47 | 0.58 | 1.00 |
| | local | 0.45 | 0.56 | 0.67 | 1.00 |
| **`small` + `vocabulary.txt`** | **all four** | **0.82** | **0.86** | **0.83** | 1.00 |

¹ Unvalidated: one judge, and that judge (Llama-3.1-8B) scored 0.78 on GroUSE, below the 0.8 bar.
Run records: `../runs/MEDIA-7B/`, `../runs/MEDIA-7B-VOCAB/`; reports: `report/`, `vocab/report/`.

### Why: names in the transcript

| Transcripts | Word error rate (3 clips) | Invented names spelt right (of 6) |
|---|---|---|
| Whisper `small` | 6 % / 15 % / 9 % | 1 (only "Velmora", and not everywhere) |
| Whisper `large-v3` (5× larger) | 8 % / 2 % / 10 % | 1 |
| Whisper `small` + `vocabulary.txt` (`hotwords`) | 2 % / 10 % / 3 % | **6** |

Without the vocabulary, 4 of the 11 answerable questions failed on exact match **with the right
passage retrieved** (recall@10 = 1.00), because the name in the passage was already wrong:
"Corrin Hill" for *Korrin Hill*, "Tomas Ferrell" for *Tomas Ferrel*, "Donhold" / "Dawnholt" for
*Dornholt*. A larger model misspells names it has never heard just the same; listing the names
fixes it. **For the team's own clips, collect the names (people, places, products) with the
clips.**

## Findings

1. **Retrieval was not the bottleneck** (recall@10 = 1.00 everywhere). With 9 documents, every
   mode sees nearly the same passages, which is why V4, naive and mix tie. Mode differences need a
   corpus large enough that retrieval can fail.
2. **The remaining errors are the 7B generator's reasoning**, identical in every mode:
   * the comparison question: it says the observatory (1998) is older than the festival (1972);
   * the two-part multi-hop question: the answer comes back garbled.
3. **`local`'s lead in the first run was noise.** It answered the comparison right once; with
   better transcripts no mode does.
4. **The judges catch what EM cannot.** "Brennan-7" for *Brennan Seven* is judged correct with
   EM = 0. This is why both are reported.
5. **RAGAS with an 8B judge is noisy.** It gave faithfulness 0.0 to the answer "1998" whose
   context states it outright. Read RAGAS numbers from a small judge as indicative only.
6. **Graph-quality check (G3) on this set reported 0/2.** Both dev-split answers are values
   ("1998", "1.2 metres"), not entities, so a check meant for entity answers cannot match them.
   The graph itself had 28 entities and 21 relations from 9 documents.

## Run 3: pictures and music (media v2)

**Setup:**
* A fourth clip, `festival_theme.wav`: an instrumental, C major at 120 BPM, no speech.
* Ingested with `--music --scenes --captioner granite3.2-vision:2b` (the laptop vision model).
* 16 questions: the 12 above, 2 whose answer is only *on screen* (title-card text in the video),
  and 2 whose answer is only in the *music* (key, tempo).
* Same models and modes, RAGAS off. Records: `../runs/MEDIA-7B-V2/`, report: `v2/report/`.

| Variant | EM | Judge accuracy | Speech / other (12) | Picture (2) | Music (2) |
|---|---|---|---|---|---|
| V4 | 0.60 | 0.63 | 9 | 0 | 1 |
| naive | 0.60 | 0.63 | 9 | 0 | 1 |
| local | 0.67 | 0.69 | 10 | 0 | 1 |
| mix | 0.53 | 0.63 | 9 | 0 | 1 |

(Judge-correct answers per question group. Judge unvalidated, as above.)

* **Music works end to end.**
  * "In which key is the theme tune?" got **"C major" in every mode**.
  * "About how many BPM?" got **"117.5 BPM"** (true value 120), judged partial. That is the
    tempo estimate's error, within its ±3 BPM tolerance; EM is 0.
  * Speech clips were labelled `speech` with "key unclear", and Whisper invented no lyrics for
    the instrumental.
* **Pictures fail at captioning, not retrieval** (recall@10 = 1.00).
  * The 2B laptop captioner read the on-screen text in 3 of 6 scenes, and never read "1.2 M" or
    "DORNHOLT WORKSHOP".
  * It invented people and faces on plain colour title cards, even though the prompt forbids
    guessing.
  * The graph modes answered "MIRROR" (partial) for the first picture question; V4 refused.
  * **Use Qwen2.5-VL-7B on a GPU** (README §6 step 10b) before drawing any conclusion about
    picture questions.
* **Extra documents cost a little.** With the 6 scene documents in the index, V4, naive and mix
  each lost one speech question they answered in run 2 (the cross-clip "Dornholt" question).
  More, noisier documents compete for the same context budget.
* **Cut detection was exact:** 6 scenes, cut at the sentence boundaries.

## Defects found by these runs (fixed)

* The RAGAS client was closed only after its event loop ended ("Event loop is closed" noise
  after each run) → closed inside the loop.
* Misspelt names had no remedy → `vocabulary.txt` / `--vocabulary` (Whisper `hotwords`).
* Scene cuts were detected in grey and missed colour changes → colour comparison.
* Drums-only audio got a key → key reported only at confidence ≥ 0.5.
* Ungated sound tags called a talk "jazz (0.88)" → tags only for clips labelled music/mixed
  (CLAP's own music/speech scores were also wrong on this audio, so they do not gate).

## Reproduce

```bash
uv run python scripts/make_media_demo.py data/raw/media_demo            # writes vocabulary.txt
uv run elevate-rag data ingest-media data/raw/media_demo --as media_demo --window-s 15 --max-window-s 25 \
  --vocabulary /dev/null        # first run: without the names (an empty vocabulary)
scripts/run_pipeline.sh configs/pipelines/local_7b.yaml configs/experiments/MEDIA_demo_local_7b.yaml \
  media_demo results/media_demo --set vector_store.backend=exact --set lightrag.vector_storage=NanoVectorDBStorage \
  --set graph_store.backend=networkx --set lightrag.graph_storage=NetworkXStorage
```

`say` output differs by a few milliseconds between runs, so regenerated clips have new checksums
and slightly different gold times; the dataset version in each run record identifies the exact
data used. These runs used `local_7b.yaml` without its second judge (`judge2` removed) to avoid a
4.9 GB download; the committed spec keeps both judges.
