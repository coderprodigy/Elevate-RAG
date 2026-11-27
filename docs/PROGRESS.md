# Progress Log

Workspace `Elevate-RAG-nova`, branch `nova`. **Nothing committed or pushed** (user instruction).

## Verified state (2026-11-26)

| Check | Command | Result |
|---|---|---|
| Install from lockfile | `uv sync --locked --extra faiss` / `uv lock --check` | consistent (198 packages, `exclude-newer 2026-11-12`) |
| Lint / format | `ruff check` / `ruff format --check` | clean (71 files) |
| Types | `mypy` (pydantic plugin) | clean (53 source files) |
| Tests | `pytest -q --cov` (with live Qdrant + Neo4j) | **126 passed, 4 skipped**, 86 % line+branch coverage |
| DB conformance | Qdrant + Neo4j live in Docker | **passed**; Milvus, pgvector, Memgraph, PG-graph not started → skipped with reason |
| Offline demo | `elevate-rag demo` | 13 variants (V0–V4, 5 LightRAG modes, R1, R3, R4) end to end, report generated, all labelled DEMO |
| Real datasets | `elevate-rag data prepare …` | all 7 prepared from pinned revisions, checksums verified (table below) |
| Model-free measurements | `scripts/model_free_measurements.py` | `results/model_free/RESULTS.md` |

### Prepared datasets

| Dataset | Docs | Questions | Dev / Test | Notes |
|---|---:|---:|---|---|
| nq | 4,839 | 500 | 100 / 400 | gold passages + distractors |
| hotpotqa | 9,801 | 1,000 | 200 / 800 | 6 duplicate paragraphs merged, gold IDs remapped |
| musique | 8,171 | 600 | 120 / 480 | 500 answerable + 100 unanswerable; answerable twins excluded |
| multihop_rag | 609 | 2,556 | 511 / 2,045 | evidence 100 % verbatim |
| graphrag_bench_medical | 1 | 2,062 | 413 / 1,649 | evidence 0.45 % verbatim → string evidence recall withheld |
| graphrag_bench_novel | 20 | 2,010 | 401 / 1,609 | upstream ID `Novel-55f0c0e2` shared by 2 questions → kept both (`#1`, `#2`) |
| ragrouter | – | 7,727 | 1,545 / 6,182 | router-only; both evidence formats normalised |

## Defects found and fixed by testing or real data

1. LightRAG in-process state leaked between indexes (shared workspace) → one workspace per index (D-20).
2. Graph-mode runs after a vector-only build silently ran without a graph → `open()` now raises.
3. A truncated JSONL tail corrupted resumed runs → tail repair before resuming (D-22).
4. The R1 rules missed "the founder/spouse/… of X" bridges → explicit rule (D-23).
5. The central figure dropped variants beyond 8 colour slots → small multiples (D-25).
6. The Hub client stalled; a DNS outage blocked offline re-preparation → verified resumable
   downloader plus cached metadata (D-26).
7. GraphRAG-Bench paraphrased evidence made string evidence recall meaningless → measured
   `evidence_verbatim_rate`, metric withheld when below 0.9.
8. RAGRouter-Bench rejected 3,414 records with `{doc_id, text}` evidence → both shapes accepted.
9. A duplicate upstream question ID in GraphRAG-Bench novel → deterministic disambiguation.

## Experiment execution status

| Block | Software | Research run |
|---|---|---|
| E1 (V0–V4) | ready; V1 retrieval measured on 4 datasets | **not run**: needs LLM (X1), judge (X2), budget (X3) |
| E2 (modes) | ready | not run (X1–X3) |
| E3 (routers) | ready; R1 and R2 classification measured on real labels | end-to-end not run (X1–X3) |
| E4 (DBs) | exact / FAISS / NetworkX benchmarked on synthetic data | server DBs not run (X4) |
| E5 / E6 | ready (specs validate) | not run (X1–X3, X6) |
| A1–A4 | ready | not run (X1–X3) |
| Human gates (G3 audit, judge kappa, failures) | export and agreement tooling ready | need two annotators (X5) |

## Session 2 (2026-11-26, after owner decisions D-27)

* Docker started; reference stack (Qdrant 1.15.5 + Neo4j 5.26.12) healthy; adapters verified live.
* Four infrastructure bugs found and fixed (see KNOWN_ISSUES).
* Ollama + sentence-transformers installed; `local_free.yaml` ($0 config) written and validated;
  20-question HotpotQA distractor subset prepared (`hotpotqa_g1`, 200 paragraphs) for gate G1.
* Model downloads stopped at owner's decision (disk 98 % full); partial download removed,
  Ollama service stopped. **No local-LLM run has been executed.**

## Session 3 (2026-11-26): ready to push and deploy

* **Run identity no longer depends on storage paths.** `config_hash()` included `data_dir`,
  `cache_dir` and `index_dir`, so moving data (e.g. to an EC2 volume) changed every run id and
  completed runs would have been re-run → location fields excluded (D-29, regression-tested).
* Storage locations come from `ELEVATE_DATA_DIR` / `ELEVATE_CACHE_DIR` / `ELEVATE_INDEX_DIR` /
  `ELEVATE_RUNS_DIR` (they were documented in `.env.example` but never read).
* Database ports bind to `127.0.0.1` by default (`ELEVATE_BIND_HOST`).
* Added `Dockerfile` (multi-stage, locked deps, non-root, tiktoken vocabulary baked in so it runs
  offline), `.dockerignore`, `.python-version`, `scripts/ec2_setup.sh`, `docs/DEPLOYMENT.md`,
  Makefile `image` / `image-smoke` / `db-ps`, and a CI job that builds the image and runs the
  demo in it. The image is **not yet built locally** (disk; see KNOWN_ISSUES).
* `elevate-rag env` no longer crashes on a hung Docker daemon.
* Checks: ruff + format clean (src, tests, scripts), mypy clean, **126 passed, 4 skipped**, 86 %.
