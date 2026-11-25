# Problems log

Every non-trivial problem found while building and running the project: what happened, the cause,
what we did, and its status. Add new entries at the top of the table. These are useful for the
report's "Limitations / threats to validity" and "Implementation details" sections.

Status: ✅ fixed · 🟡 mitigated / workaround · 🔴 open

| # | Date | Area | Problem | Cause | Resolution | Status |
|---|---|---|---|---|---|---|
| P20 | 11-25 | Planner | Graph pipelines loaded the embedding model (17 s) before discovering the graph was not built | Model loading came before the existence check | Check the graph manifest first; planning a blocked run now takes milliseconds | ✅ |
| P19 | 11-25 | Dry runs | Pricing a V4 run did real retrieval + reranking (hours locally for long chunks) | `estimate_cost` retrieved to price the exact prompt | Exact price only when retrieval is cached; otherwise a full-budget upper bound (instant) | ✅ |
| P18 | 11-25 | Graph / Claude | LightRAG asks for `response_format=json_object` (keywords); the Claude API has no such parameter | Provider difference | Adapter drops it and adds "reply with a single JSON object" to the system prompt; **verify on the first live sample build** | 🟡 verify live |
| P17 | 11-25 | Dependencies | Plan pinned LightRAG v1.5.7; v1.5.8 is current | Release after the design doc | Use 1.5.8 (locked in `uv.lock`); API checked by probing (`ainsert` + `chunking_func`, `aquery_data`, roles) | ✅ |
| P16 | 11-25 | Graph retrieval | `global` mode sometimes returns status "Query returned no results" | No relation matched the query's high-level keywords | Treated as an empty context (generator refuses), not an error | ✅ |
| P15 | 11-25 | Graph / LightRAG | `SharedNamespaceBackingConflictError` when two graphs are open in one process | LightRAG keeps process-wide namespaces per *workspace* | One LightRAG workspace per graph location under `data/graphs/`; pipelines close storages after each run (`aclose`) | ✅ |
| P14 | 11-25 | Retrieval eval | `R@10` column equals `R@8` | Searcher returns only `final_k` = 8 passages | Documented in the report; R@k is meaningful for k ≤ 8 | 🟡 |
| P13 | 11-25 | Tooling | A background run used `timeout`, which macOS lacks; the run silently did nothing | GNU coreutils not installed | Don't use `timeout` in scripts; rely on the tool's own timeouts | ✅ |
| P12 | 11-25 | Local compute | A retrieval run stalled overnight | The Mac went to sleep during a GPU job | Long local jobs run under `caffeinate -i` | ✅ |
| P11 | 11-24 | Local compute | Qwen3-Embedding build crashed (exit 139, segfault) | Ran alongside the reranker on the 16 GB GPU (MPS memory pressure); runs fine alone | Never run two GPU jobs at once; rerun alone (resumable) | 🟡 |
| P10 | 11-24 | Local compute | Reranker (bge-reranker-v2-m3) takes ~15 s/question on 50 × 500-token chunks (MPS; CPU 25 s; fp16 only −20%) | 568M-parameter cross-encoder on a laptop GPU | `rerank_candidates` setting (default 50 = plan, recorded in run configs); large V4 runs moved to the Waterloo GPU | 🟡 |
| P9 | 11-24 | Concurrency | Random `sqlite3.InterfaceError` / `None` cache reads during V4 runs | Retrieval worker threads shared one SQLite connection without a lock | Lock inside `ResponseCache` and `UsageLedger`; regression test with 16 threads | ✅ |
| P8 | 11-24 | Chunking | Sentence splitter dropped closing quotes (`"Then she came.` lost its `"`) | Quote was consumed by the split pattern | Quote moved into a look-behind; unit test | ✅ |
| P7 | 11-24 | Cost | Dry run priced `--repeat` (uncached) runs at $0 | Estimate always assumed cache hits | `estimate_cost(use_cache=…)`; test | ✅ |
| P6 | 11-24 | Gold labels | GraphRAG-Bench evidence maps to chunks for only ~2% of questions | Evidence sentences are LLM paraphrases, not quotes (median fuzzy match 71/100) | Score GraphRAG-Bench retrieval with LLM-checked **evidence recall** (the benchmark's own metric) | 🟡 needs judge key |
| P5 | 11-24 | Metric choice | NQ recall@k looked very low (BM25 R@5 = 0.22) | NQ questions have ~4 gold passages; recall counts all of them | Added hit@k (DPR top-k accuracy), the standard NQ metric | ✅ |
| P4 | 11-24 | Data leakage | 2,499 RAGRouter-Bench questions are also in our evaluation sets (all 1,896 medical + 603 MuSiQue) | RAGRouter-Bench is built from MuSiQue and GraphRAG-Bench medical | Overlaps are dropped at build time; R2 trains on 3 domains | ✅ |
| P3 | 11-24 | Data validity | 218 MuSiQue "unanswerable" questions would become answerable in a pooled corpus | Their missing evidence appears in other questions' paragraphs | Kept only unanswerable questions whose missing evidence stays out of the pool | ✅ |
| P2 | 11-24 | Determinism | Claude Haiku/Sonnet 5.5 reject `temperature`; the API has no seed | API change in the 5.5 models | Default sampling + response cache; measured run-to-run variance (`--repeat`, `eval variance`) | 🟡 run pending |
| P1 | 11-24 | Auth | Every Hugging Face download failed with HTTP 401 | Expired `huggingface-cli` token (2024) cached on the Mac and sent automatically | Code uses `HF_TOKEN` or anonymous access only (`HF_HUB_DISABLE_IMPLICIT_TOKEN`); optional `huggingface-cli logout` | ✅ |
