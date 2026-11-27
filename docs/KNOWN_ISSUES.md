# Known Issues, Limitations and Blockers

Severity: **B** = blocks research results · **M** = limits a requirement · **L** = minor.

## External blockers (need people, credentials, services or budget)

| # | Sev | Issue | What unblocks it |
|---|---|---|---|
| X1 | B | No LLM credentials or model server in this environment (`ANTHROPIC_API_KEY`, `OPENAI_COMPAT_BASE_URL` unset). Graph extraction, generation, LLM routing and judging have run **only with offline stand-ins**. | Team provides keys / a vLLM endpoint; set them in `.env`. |
| X2 | B | **Judge model not chosen.** The proposal's generators span Claude, Gemma and Qwen, so the judge must come from another family. No ID is shipped (D-15); `experiment validate` fails until `models.judge` is set. | Team decision at kickoff. |
| X3 | B | **Spending cap is $0.** Every paid call is blocked before it is made. | Run `index pilot` (5 %), estimate, agree a cap, set `budget.max_usd`. |
| X4 | M | **Qdrant and Neo4j adapters verified against live containers** (reference stack). Milvus, pgvector, Memgraph and PostgreSQL-graph were not started (disk space): implemented, not externally verified; their tests skip with a reason. | `make db-up PROFILE=all`, set `ELEVATE_*` URLs, run `pytest tests/integration/test_db_services.py`. |
| X5 | B | Human annotation gates not passed: 50-triple audit (G3), 100-answer judge validation (kappa ≥ 0.6), 50 failures per system. The tooling exports blank CSVs; **no labels have been fabricated**. | Two team members label the exported CSVs. |
| X6 | B | Hardware/disk: 16 GB RAM and only ~4–5 GB free disk (98 % full; another Docker project shares the machine). The $0 local plan (`configs/pipelines/local_free.yaml`: Qwen2.5-3B generator, Llama-3.2-3B judge, bge-small + bge-reranker) is configured but **not executed**: model downloads were stopped by owner decision when the disk could not hold them. Earlier note: The mid-tier generator Qwen3.6-35B-A3B and Qwen3-Embedding-8B cannot run locally; even the reference embedder (Qwen3-Embedding-0.6B, ≈1.2 GB) and `sentence-transformers`/torch (≈1 GB) were not installed, so the `local-models` extra is **implemented but not externally verified**. | GPU host or bigger workstation; `uv sync --extra local-models`. |

## Partially implemented / deviations

| # | Sev | Item | Detail |
|---|---|---|---|
| P1 | M | FR-G11 LlamaIndex `SchemaLLMPathExtractor` fallback | **Not implemented.** `index validate` reports gate G3 and prints the fallback instruction. The integration point exists (`LightRAG.ainsert_custom_kg` accepts externally extracted triples with source chunks), but adding llama-index (heavy) without credentials to verify it was judged not worth an unverifiable code path. |
| P2 | M | RAGAS package | Metrics implemented natively to the RAGAS definitions (D-19); the `ragas` package itself is an optional extra and has not been run. |
| P3 | M | GraphRAG-Bench official scorers | Not wired in. Its gold evidence is **paraphrased** (0.5 % verbatim, measured), so string-match evidence recall is reported as *undefined* for it (`evidence_verbatim_rate` < 0.9 in the manifest). Its LLM-judged evidence recall / context relevance need the benchmark's scripts plus an LLM. |
| P4 | M | External reference systems (MS GraphRAG, HippoRAG 2, LlamaIndex PGI, Neo4j GraphRAG) | Integrated through the predictions contract (`experiment external`, D-14); the adapters that *run* those systems are not included. |
| P5 | L | Langfuse tracing | Not implemented. Every call is recorded in the run ledger (`ledger.json`) and per-question `calls`, which carries the same information; Langfuse export is future work. |
| P6 | L | R2 "small encoder" variant (optional in the plan) | Not implemented; TF-IDF + SVM is. |
| P7 | L | E4 Level B warm-up | Level A applies 1,000 warm-up queries and median-of-3; Level B end-to-end runs report per-query latency percentiles without a separate warm-up phase. |
| P8 | L | Community layer | Leiden (NetworkX 3.7) + LLM reports; at query time the top reports are added under the shared budget instead of MS GraphRAG's full map-reduce (documented in `graph/community.py`). |
| P9 | L | Temperature 0 | Not available for current Claude models (API rejects sampling params); recorded per call as `sampling_controls: unsupported` (D-09). Reproducibility relies on the response cache. |

## Observations from real-data measurements (not defects)

* The R1 keyword rules reach 97 % type accuracy on NQ but only ≈45–52 % on HotpotQA, MultiHop-RAG
  and GraphRAG-Bench (test splits). This motivates R2/R3. The rules were **not** tuned on these
  test results.
* LightRAG 1.5.7 shares in-process state per *workspace*; fixed by one workspace per index (D-20).
* The Hugging Face hub client stalled on this network; datasets are fetched by a resumable HTTPS
  downloader verified against the Hub's LFS SHA-256 / git blob SHA-1. The network also dropped
  mid-session (DNS failure); downloads resume from `.part` files, and verified file metadata is
  cached so re-preparing works offline (D-26).

## Bugs found while bringing up real services (fixed, regression-tested)

* Neo4j heap + page cache exceeded the container memory limit → sized from `.env` (compose).
* `qdrant-client` 1.20 is incompatible with the pinned server 1.15.5 → client pinned `<1.16`.
* Importing `pymilvus` loads `./.env` into `os.environ`, masking missing configuration; Milvus
  connection errors surfaced as raw gRPC errors → URL read before import, errors normalised to
  `StorageError` (`test_milvus_unreachable_raises_storage_error`).
* DB conformance tests now *skip* (with reason) when a configured service is not reachable.

## Deployment readiness (session 3)

* **The Docker image has not been built locally**: the disk had ~3 GB free and the Docker daemon
  was unresponsive. Base-image tags were verified to exist, the dependency set resolves from the
  lockfile (`uv sync --dry-run`: 109 packages, all DB clients, no dev tools, no torch), and the
  compose file validates. The CI `docker-image` job builds the image and runs the offline demo in
  it with no network: that is the first real verification, on the first push.
* `elevate-rag env` crashed when the Docker daemon hung (`docker info` timeout) → now reports
  `docker_daemon: false` (`test_env_reports_a_hung_docker_daemon_instead_of_crashing`).
* No `LICENSE` file: `pyproject.toml` declares MIT, but the copyright holder must be chosen by the
  project owner before publishing.

## Found in gate G1 (session 4)

* **Docker VM disk failing** (`Input/output error` creating a Qdrant collection; Neo4j handshake
  hangs; `docker ps` hangs). Likely damaged when the host disk filled. Restart Docker Desktop
  when no other project is using it, then re-run `make db-up` and the live DB tests.
* **3B judge unreliable**: accepted clearly wrong answers (5 of 6 inspected for `mix`). No
  judge-based number counts until the human agreement gate (κ ≥ 0.6) passes.
* **3B extractor too weak for RQ2**: 60 % of entities have no relation; 1,263 malformed
  relation lines discarded by LightRAG. Use a stronger extractor for E-block runs.
* **Latency under caching**: queries with cached LLM calls are excluded from latency
  (`latency_n`, `cache_hit_rate` in summaries). For latency comparisons run each variant against
  an empty LLM cache (fresh `ELEVATE_CACHE_DIR`).
* **macOS + FAISS + local torch models** cannot share a process (OpenMP); refused with advice.
* **Ollama context**: must be started with `OLLAMA_CONTEXT_LENGTH=8192` (or more).
* **Idle sleep** pauses long local runs on a laptop; wrap them in `caffeinate -i`.

## Found in the 7B G1 run (session 5)

* The 8B judge is much more consistent than the 3B one (judge-free check agrees on all but one
  inspected item) but is **still unvalidated** against humans: the κ gate is open.
* The 7B G1 graph was built before the spacing-insensitive entity key (D-35): it contains 84
  split duplicate nodes (duplicate rate 4.8 %). Rebuild before reusing it for anything but G1.
* Graph modes with a local 7B model are slow (local p50 116 s vs 14 s for V4 on a laptop).
* On 20 questions no retrieval mode differs significantly from V4; RQ2 needs the full E2.

## Cluster readiness (session 5)

* **The SLURM / vLLM / Apptainer path has not run on a real cluster or GPU.** Checked: Bash
  syntax, dry-run submission (job graph, arrays, dependencies), the vLLM start/wait/stop logic
  with a stub server, config resolution and spec validation for both cluster configs, and that
  every model id/revision exists and is ungated on Hugging Face. Tier S0 (`submit.sh smoke`) is
  the first real test; expect site-specific edits in `infrastructure/cluster/env.sh` (partition
  names, account, module loads, GPU type).
* **Qwen3.6-35B-A3B needs ~72 GB of weights** (2×80 GB or 4×48 GB GPUs). Smaller clusters use
  `cluster_gpu_7b.yaml` (Qwen2.5-7B, one 24 GB+ GPU).
* **Model download ≈ 105 GB** for `cluster_gpu.yaml`; put `HF_HOME` on scratch/project storage.
* **Apptainer `--fakeroot`** may be disabled on some clusters; use the uv venv instead (no root).
* `results/g1` (3B) predates the `extends` fix (D-31) and the run-id change (D-30): re-running it
  creates new run ids and index keys; model inputs are unchanged.
* Old partial indexes from the failed Qdrant/FAISS G1 attempts remain under `indexes/hotpotqa_g1/`
  (git-ignored, a few MB). They are not used; delete them if you want the space.

## Hygiene

* Nothing has been committed or pushed (explicit user instruction, D-17).
* `.env` is git-ignored. It was created in session 2 with generated local-database passwords only
  (no API keys); `.env.example` contains no secrets. A scan of every pushable file found no keys,
  passwords or personal paths.
* `data/`, `indexes/`, `runs/`, `demo_output/` and caches are git-ignored.
