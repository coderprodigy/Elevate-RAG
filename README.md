# Elevate-RAG: Hybrid Vector + Graph RAG

**When does graph structure pay for itself in retrieval-augmented generation, and can a router
capture that benefit automatically?** (ECE 750 group project, University of Waterloo, Fall 2026)

This repository builds one shared index for **vector retrieval** (BM25, dense, hybrid + reranker)
and **graph retrieval** ([LightRAG](https://github.com/HKUDS/LightRAG)), puts a **query router**
on top, and measures accuracy, cost and latency across six datasets: Natural Questions,
HotpotQA, MuSiQue, MultiHop-RAG and GraphRAG-Bench (novel + medical). RAGRouter-Bench is used to
train the router.

- Design and phase plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- What is still open: [docs/PENDING.md](docs/PENDING.md) · Problems log: [docs/PROBLEMS.md](docs/PROBLEMS.md)
- Detailed setup guides: [docs/setup/](docs/setup/README.md)

Every model call goes through one metered gateway (response cache, usage ledger, budget caps), so
`hgrag usage` always shows the true API spend and nothing is paid for twice.

---

## 1. Requirements

| Need | For | Notes |
|---|---|---|
| macOS or Linux, Python **3.12** | everything | `uv` installs the right Python if needed |
| [**uv**](https://docs.astral.sh/uv/) | dependencies | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| ~25 GB free disk | datasets, indexes, model weights | |
| GPU (Apple MPS or CUDA) | embeddings and reranking | CPU works but is slow; see §5 |
| API keys | answering, judging, graph building | see §3; **not** needed for setup, data or indexes |
| Docker *(optional)* | database benchmarks (Phase 6) | see [docker/README.md](docker/README.md) |

## 2. Install

```bash
git clone https://github.com/coderprodigy/Elevate-RAG.git
cd Elevate-RAG
make setup                        # uv sync (creates .venv) + git pre-commit hooks
make check                        # lint + type check + unit tests, all offline (~10 s)
uv run hgrag --help               # the CLI: data, index, eval, graph, router, bench, exp, report
```

`make check` needs no keys, network or GPU. If it passes, the install is good.
Optional database clients for the benchmarks: `uv sync --group bench`.

## 3. API keys

```bash
cp .env.example .env              # .env is git-ignored: never commit it
```

Fill in what you need for the work you are doing:

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | generation (Claude Haiku/Sonnet 5.5), graph extraction, LLM router |
| `OPENAI_API_KEY` *or* `GEMINI_API_KEY` | the LLM judge (must be a different model family from every generator) |
| `VLLM_BASE_URL`, `VLLM_API_KEY` | open-weight models served on the GPU server ([docs/setup/05](docs/setup/05_waterloo_servers.md)) |
| `HF_TOKEN` *(optional)* | faster Hugging Face downloads; anonymous access works |
| database variables | only for Phase 6 benchmarks (defaults match `docker/`) |

Check a key with a tiny call (< $0.001), then see the spend:

```bash
uv run hgrag llm ping
uv run hgrag usage
```

Spending caps live in [configs/budget.yaml](configs/budget.yaml). A run that would exceed a cap
stops cleanly and can be resumed. Runs estimated above the confirm threshold need `--yes`.

## 4. Data (free, about 5 minutes)

Raw and processed data are **not** in the repository (licences and size); everyone builds them
locally with three commands. All sources download automatically:

```bash
uv run hgrag data download        # ~400 MB into data/raw/ (Hugging Face + DPR)
uv run hgrag data build           # data/processed/<dataset>/ in ~30 s
uv run hgrag data card            # summary table + reports/data_cards.md
```

| Dataset | Questions used | Corpus |
|---|---|---|
| `nq` | 500 | gold passages + 10 hard negatives each |
| `hotpotqa` | 1,000 (distractor, hard) | 10 paragraphs per question, pooled |
| `musique` | 500 answerable + 100 unanswerable | paragraphs, pooled |
| `multihop_rag` | 2,556 | 609 news articles |
| `graphrag_bench_novel` / `_medical` | 2,009 / 2,062 | 20 novels / 1 medical text |
| `ragrouter_bench` | 5,228 (router training only) | none |

**The dev/test splits are frozen and committed** (`data/processed/*/splits.json`), so everyone
evaluates on identical questions. `hgrag data build` refuses to run if the sampled questions no
longer match the split. **Do not delete or edit these files.** Details and licences:
[docs/setup/03_datasets.md](docs/setup/03_datasets.md).

## 5. Retrieval indexes (free, local)

```bash
uv run hgrag index build --no-embeddings     # chunks, gold-chunk maps, BM25 (~2-3 min, CPU)
uv run hgrag index build -e bge-small        # + small embedder and HNSW (~10 min on Apple MPS)
uv run hgrag index build -e qwen3-emb-0.6b   # + reference embedder (slower; run alone)
uv run hgrag index evaluate                  # V1-V4 retrieval recall on dev, no LLM calls
```

Index builds resume: parts already built are skipped. On a laptop, **run one GPU job at a time**
and keep the machine awake for long jobs (`caffeinate -i …` on macOS). The cross-encoder reranker
is slow on long chunks locally (about 15 s per question); large V4 runs belong on a GPU server.

## 6. Graphs (paid: check the cost first)

```bash
uv run hgrag graph estimate                       # upper bound per dataset, no calls
uv run hgrag graph build --sample -d multihop_rag # index 5% of chunks, measure the real cost
uv run hgrag graph build -d multihop_rag          # full build (resumable)
uv run hgrag graph quality                        # answer-entity coverage (gate: ≥ 60%), graph shape
```

With Claude Haiku 5.5 as the extractor, all six graphs together cost at most about $38 (upper
bound); the sample build prints the extrapolated real cost.

## 7. Running experiments

```bash
uv run hgrag exp list                           # all experiments (configs/experiments/)
uv run hgrag exp plan --exp g1_smoke            # every run: pending questions, cost bound, blockers
uv run hgrag exp run  --exp g1_smoke            # resumable; asks for --yes above the threshold
uv run hgrag eval judge --experiment g1_smoke   # LLM judge (primary metric)
uv run hgrag router evaluate --experiment e2_modes
uv run hgrag report build                       # all tables + figures from runs/ (no calls)
```

Rules the code enforces: every pipeline shares the same chunks, prompt, reranker and context
budget, so only retrieval differs; the **test split is locked** until an experiment is marked
`frozen: true` at a gate review; `--dry-run` / `exp plan` always show the cost before spending.

## 8. Repository layout

```
src/hgrag/      data · index · retrieval · graph · pipelines · router · eval · bench · analysis · llm
configs/        models + prices, budget, datasets, retrieval, graph, router, experiments/
data/           only the frozen splits are tracked; everything else is built locally
docker/         database stacks for the benchmarks
docs/           setup guides, pending items, problems log
reports/        generated tables and figures for the final report
tests/          offline unit tests (+ integration tests: `make test-int`)
```

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Hugging Face `401` on public downloads | An expired token saved by `huggingface-cli login`. The code ignores it, but other tools may not: `huggingface-cli logout`, or set a fresh `HF_TOKEN` in `.env`. |
| `role 'generator' has no model yet` | Pass `--model claude-haiku-5-5`, or set the role in `configs/models.yaml`. |
| Crash (exit 139) during embedding | Two GPU jobs at once ran out of memory. Run one at a time; builds resume. |
| A run stopped with "cap would be exceeded" | Budget guard. Raise the cap in `configs/budget.yaml` if intended, then rerun: it resumes. |
| Paths with spaces | Quote them in shell commands. |

## Licence

Code: [MIT](LICENSE). Datasets keep their own licences (recorded in `configs/datasets.yaml`) and
are never redistributed here.
