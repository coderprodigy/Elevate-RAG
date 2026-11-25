# Pending items by phase

Single source of truth for everything still open. Update it at the end of every implementation step.

Updated: 2026-11-25, all phases implemented in code; runs pending.
Owners: **You** = Dilreet · **Team** = group members · **Claude** = implementation.
☐ open · ☑ done · *(optional)* = not required for the Must tier.

---

## A. Blocking right now

| # | Item | Owner | Blocks | How |
|---|---|---|---|---|
| A1 | Put `ANTHROPIC_API_KEY` in `.env`, then run `uv run hgrag llm ping` and `uv run hgrag usage` (< $0.001) | You | Phase 0 live check; every paid run | `cp .env.example .env` · [02 §2](setup/02_accounts_api_keys_and_models.md) |
| A2 | Agree real spending caps. Current placeholders: $200 total, $25 per run, $2 confirm threshold | You + Team | Any paid run above $2 | Edit [configs/budget.yaml](../configs/budget.yaml) |
| A3 | Choose the **judge** model (a GPT or Gemini model; it must differ from every generator family) and add `OPENAI_API_KEY` or `GEMINI_API_KEY` | You + Team | Phase 2 judge metric; G1 | [02 §1, §3](setup/02_accounts_api_keys_and_models.md) |

## B. Project-wide (no single phase)

### B1. Model freeze ("week 1": do before the G1 smoke run)
Fill in [02 §3](setup/02_accounts_api_keys_and_models.md). I then add the entries to [configs/models.yaml](../configs/models.yaml).

- ☐ Judge: exact model ID + price (same as A3). — You + Team
- ☐ Reference generator (mid open): Qwen3.6-35B-A3B or DeepSeek V4-Flash; exact HF ID + quantisation on the Waterloo server. — You + Team, finalised on day 4
- ☐ Small open generator: Gemma 4 12B; exact HF ID. — You + Team
- ☐ Open extractor: Qwen3.6-35B-A3B; exact ID (E5 comparison). — You + Team
- ☐ API embedder for E5: OpenAI text-embedding-3-large / Voyage / Gemini; pick one. — Team *(Should tier)*
- ☐ API reranker for E5: Cohere or Jina; pick one. — Team *(Should tier)*
- ☐ Confirm Claude Haiku 5.5 / Sonnet 5.5 prices on Anthropic's pricing page on freeze day (now $0.10/$0.50 and $2/$10 per MTok). — You

### B2. Accounts and keys (beyond A1/A3)
- ☐ `OPENROUTER_API_KEY`: only if the Waterloo server is delayed (fallback host for open models). — You *(conditional)*
- ☐ `COHERE_API_KEY` or `JINA_API_KEY`: API reranker, E5. — You *(Should tier)*
- ☐ `VOYAGE_API_KEY`: API embedder, E5. — You *(optional)*
- ☐ `HF_TOKEN`: faster downloads only; anonymous access works. — You *(optional)*
- ☐ Langfuse keys: tracing only; the ledger already tracks spend. — You *(optional)*
- ☐ Set a monthly spending limit in each provider console as a hard backstop. — You

### B3. Team and process
- ☐ Assign the five role owners (Data/Eval, Vector RAG, Graph RAG, Router, Infra). See plan §4 and the design doc's roles table. — You
- ☐ Line up **two labellers**: 100 answers for judge kappa (Phase 3, ~2–3 h each) and 50 triples for the graph audit (Phase 4, ~1 h each). — You
- ☐ Gate sign-offs, a 15-minute team review each: **G1** data + harness (day 2) · **G2** V4 locked (day 4) · **G3** graphs pass checks (day 6) · **G4** results frozen (~day 14). — Team

### B4. Repository and machine
- ☐ First git commit. **On hold at your request.** — You decide when
- ☐ GitHub remote: create an empty private repo and send me the URL. — You *(if the team shares code)*
- ☐ Remove the expired 2024 Hugging Face token: `huggingface-cli logout`. Only affects other tools. — You *(optional)*
- ☐ Docker runtime on the Mac (OrbStack). The main plan runs databases on the Waterloo server. — You *(optional)*
- ☐ Ollama on the Mac for local smoke tests of Gemma. — You *(optional)*

---

## Phase 0 — Foundations ☑ code done
- ☐ Live API ping (A1). — You
- ☐ Langfuse tracing hook. — Claude *(optional)*

## Phase 1 — Data layer ☑ code done, data built and frozen
- ☐ Review [reports/data_cards.md](../reports/data_cards.md) and sign off as part of G1. — You / Data lead
- ☐ Check the MultiHop-RAG and GraphRAG-Bench licences before sharing anything outside the team. — Data lead
- ☐ Decide whether to swap the MuSiQue Hugging Face mirror for the official Google Drive files (the build is identical in format). — Data lead *(optional)*
- ☐ UltraDomain loader for pairwise "global" judging. — Claude *(Could tier)*

## Phase 2 — Evaluation harness + V0 ⏳ in progress · Claude
- ☑ **2a (done):** pipeline interface; V0 closed-book; resumable runner (`--limit`, `--dry-run`, test-split lock, confirm threshold); EM/F1, retrieval, refusal and efficiency metrics; statistics (bootstrap, McNemar, paired bootstrap, Holm, MDE); `hgrag eval run|score`.
- ☑ **2b (done):** LLM judge (correct/partial/wrong); pairwise judge with order swap; faithfulness + GraphRAG-Bench evidence recall (LLM-checked, through the gateway); router metrics; human-labelling CLI + Cohen's kappa; `hgrag eval judge|context-metrics|label|kappa`.
- ☐ **G1 smoke run:** V0 on 20 dev questions × 6 evaluation datasets. Dry run: upper bound **$0.017** on Haiku. **Needs A1** (A3 only for the judge part). Command: `uv run hgrag eval run --pipeline v0 --experiment g1_smoke --limit 20 --model claude-haiku-5-5`
- ☐ Power/MDE check on the V0 dev results. Decides whether NQ/MuSiQue samples should grow. — Claude + Eval lead
- ☐ Live judge check: `hgrag eval judge --experiment g1_smoke` (needs A3).

## Phase 3 — Vector baseline V1–V4 · code ☑ · runs pending
- ☑ Chunkers (sentence 512/64, 256, 1024; paragraph), content-hash chunk IDs, gold-chunk maps (MultiHop-RAG evidence mapped 100%).
- ☑ BM25, FAISS HNSW (M=32, efC=200, efS=128), RRF (k=60), cross-encoder reranker, 8k-token context budget, RAG prompt with citations + refusal; pipelines v1–v4; retrieval cache.
- ☑ Free retrieval-only evaluation (`hgrag index evaluate`), with hit@k for NQ.
- ☐ Embeddings: bge-small (dev) and Qwen3-Embedding-0.6B (reference) for all datasets. — running locally in background; rerun `uv run hgrag index build -e qwen3-emb-0.6b` if interrupted (resumes)
- ☐ Retrieval tuning on dev (free): chunk size (sent256/512/1024 on MultiHop-RAG), `candidates`, `final_k`. — Claude, after embeddings finish
- ☐ V1–V4 generation runs on dev. **Needs A1 + a generator** (Haiku 5.5 stand-in until day 4) and your OK after the dry run.
- ☐ Judge validation: 2 labellers × 100 answers (`hgrag eval label`), then `hgrag eval kappa` ≥ 0.6. — **Team**, needs A3
- ☐ Variance check: 50 questions × 3 uncached runs (Claude cannot use temperature 0): `hgrag eval run ... --limit 50 --repeat 1` (then 2, 3), then `hgrag eval variance`. Code ☑; **run needs A1**.
- ☐ GraphRAG-Bench retrieval: evidence is paraphrased (only ~2% string-mappable), so it is scored with LLM-checked **evidence recall** (`hgrag eval context-metrics --metric evidence_recall`), as the benchmark intends. — needs A3
- ☐ Freeze V4 config at gate **G2** (`frozen: true` in configs/experiments/e1_baselines.yaml), then E1 test runs.
- ☐ Qdrant adapter (reference vector DB) — moved to day 4 / Phase 6, with the database stack on the server.
- ☐ **Run V4 reranking at scale on the Waterloo GPU.** On this Mac the reference reranker (bge-reranker-v2-m3) needs ~15 s per question for 50 long (~500-token) chunks (MPS; CPU ~25 s; fp16 only ~20% faster). MultiHop-RAG test alone would take ~8.5 h locally. Short-chunk datasets (NQ, HotpotQA, MuSiQue) are fine locally. `rerank_candidates` (default 50 = plan) can be lowered for local dev only; it is recorded in every run config. — day 4
- ☐ Keep the Mac awake for long local jobs (`caffeinate -i …`); a sleeping Mac stalled one run overnight. — You

## Day 4 — Waterloo server setup (pause point) · You + Claude
- ☐ Access: VPN (off-campus), SSH key, a GPU usable for several hours. — You
- ☐ Run [scripts/cluster/probe_server.sh](../scripts/cluster/probe_server.sh) on the server and send me the output. — You
- ☐ Answer [05 §2](setup/05_waterloo_servers.md): host, scheduler, containers, scratch disk. — You
- ☐ vLLM launch scripts, SSH tunnel, `waterloo_vllm` model entries, Qdrant/Neo4j on the server. — Claude

## Phase 4 — Graph layer (LightRAG) · code ☑ · builds pending
- ☑ Adapter, cost-first build (`hgrag graph estimate` → `graph build --sample` → `graph build`), graph-mode pipelines, quality checks, triple audit CLI.
- ☐ **Live sample builds** (≈ $1.91 for all six 5% samples on Haiku). **Needs A1.** Confirms P18 (JSON keyword replies from Claude).
- ☐ Full builds after approval (upper bound ≤ $38 total on Haiku; measured extrapolation printed by the sample build).
- ☐ Build graphs with the reference embedder (Qwen3) once its embeddings are done; bge-small graphs are fine for dev.

### Phase 4 (original checklist)
- ☐ LightRAG adapter fed with our chunks; chunk-ID equality test.
- ☐ 5% extraction sample → cost estimate → **your approval** → full indexing (Haiku: roughly $5–15 per dataset).
- ☐ Graph quality: answer-entity coverage ≥ 60%, graph shape, triple audit (**Team**, 50 triples).
- ☐ E2 runs on dev and test.
- ☐ LlamaIndex fallback only if coverage < 60%; community layer; Microsoft GraphRAG / HippoRAG 2 references. — *(Could tier)*

## Phase 5 — Router · code ☑
- ☑ R0–R4, oracle, both label schemes, LODO, RAGRouter-Bench CV (TF-IDF: acc 0.964 / macro-F1 0.963).
- ☐ E3 needs E2 per-mode runs on **dev and test** (`v4`, `lr_local`, `lr_global`, `lr_mix`), then `hgrag router evaluate --experiment e2_modes`. **Needs A1 + graphs built.**
- ☐ R3 (`--llm --yes`) one Haiku call per test question (cents).

### Phase 5 (original checklist)
- ☐ R0–R4, oracle, both label schemes, leave-one-dataset-out, RAGRouter-Bench 5-fold CV (3 domains after deduplication).
- **Needs:** budget approval for R3 calls (cheap).

## Phase 6 — Databases (E4) · code ☑ · runs pending
- ☑ Compose stacks, vector + graph adapters, Level A harnesses, Level B via `--graph-backend`, `hgrag bench`.
- ☐ Bring up stacks on the Waterloo server; `hgrag bench health`; check pinned image tags still exist (docker/README.md). — day 4
- ☐ Run `hgrag bench vector -d multihop_rag -s faiss-flat -s faiss-hnsw -s qdrant -s pgvector -s milvus` and `hgrag bench graph -s networkx -s neo4j -s memgraph -s postgres` (free; needs graphs for the real graph, or `--synthetic 100000`).
- ☐ Server-store adapters (Qdrant, pgvector, Milvus, Neo4j/Memgraph, Postgres) are written against current client APIs but untested until the services run: `make test-int`.

### Phase 6 (original checklist)
- ☐ Compose files with pinned images and resource limits; Level A vector and graph benchmarks; Level B swaps.
- **Needs:** the container answer (Docker / Podman / Apptainer) from the day-4 probe.
- ☐ Weaviate / Chroma / Redis through VectorDBBench. — *(optional)*

## Phase 7 — Model comparison + ablations · code ☑
- ☑ Experiment planner + YAMLs for E1, E2, E4, E5, E6, A1, A3; `hgrag exp plan` shows costs and blockers.
- ☐ Add the `@waterloo` model keys to configs/models.yaml on day 4 (E5/E6 arms are blocked until then).
- ☐ A1 needs `hgrag index build -d multihop_rag --chunker sent256` (and `sent1024`) first.
- ☐ A2 (entity-merge threshold) and A4 (community layer): not implemented (Could tier).

### Phase 7 (original checklist)
- ☐ Experiment planner with dry-run costs; E5, E6, A1–A4.
- **Needs:** B1 choices, B2 keys, and approval of each experiment's dry-run cost.

## Phase 8 — Analysis + report · code ☑
- ☑ `hgrag report build|compare|errors-sample|errors-tag`, validated palette, figures checked visually.
- ☐ After G4: `hgrag report errors-sample --exp e2_modes` then two people run `errors-tag` (≈ 1 h each). — **Team**

### Phase 8 (original checklist)
- ☐ Statistics pass, error-analysis tagging (50 failures per system), figures, appendix.
- ☐ Report writing (days 15–20).

---

## Decisions already made (for reference)
- Plan §1.2 issues 4–11 resolved as proposed. Open models and databases run on the Waterloo servers. About 1 week of build within ~20 days.
- No server-side model fallback for Claude refusals: they are logged, so models never swap mid-experiment.
- RAGRouter-Bench overlap removal leaves 3 router domains (medical is fully dropped).
- GraphRAG-Bench corpus is small enough (about 1.5M tokens total), so no corpus cap.
- Faithfulness is computed with our own RAGAS-style implementation through the gateway, not the `ragas` package. That keeps every call metered and cached and avoids the LangChain dependency.
