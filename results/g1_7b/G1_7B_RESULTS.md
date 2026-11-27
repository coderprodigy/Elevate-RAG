# Gate G1 with mid-size local models (2026-11-27)

**Verdict: the pipeline passes G1 again, and the 7B/8B pair fixes most of what made the 3B run
unusable.** The judge now agrees closely with a judge-free check, the graph is 2.4× denser, and
answer quality roughly doubles. Retrieval modes still cannot be told apart on 20 questions: no
paired difference is significant, and by the judge-free check all six variants answer 11–12 of
20. These are "mid-size local models" numbers and answer no research question.

## Setup

Same as [`../g1/G1_RESULTS.md`](../g1/G1_RESULTS.md) (20 HotpotQA distractor questions, 200
paragraphs, bge embedder and reranker, exact vectors + NetworkX, same prompts), except the LLMs:

| Role | 3B run | This run |
|---|---|---|
| generator / extractor / keywords / router | Qwen2.5-3B-Instruct | **Qwen2.5-7B-Instruct** |
| judge (different family) | Llama-3.2-3B | **Llama-3.1-8B** |
| extraction output cap | 1,536 tokens | 2,048 tokens |

Config `configs/pipelines/local_7b.yaml`, spec `configs/experiments/G1_smoke_local_7b.yaml`, run
with `scripts/run_pipeline.sh` (command in [docs/RUNNING.md](../../docs/RUNNING.md) §2). Cost
$0.00. Graph build 194 min (209 extraction calls, 0 errors), six variants 62 min, on a 16 GB
Apple-silicon laptop. Ollama context 8192: no prompt truncation during this run.

## Results (n = 20; one question = 5 points)

| Variant | Judge accuracy (95 % CI) | Gold answer in response¹ | EM | F1 | Recall@10 | Latency p50² |
|---|---|---|---|---|---|---|
| V4 (vector reference) | 0.50 [0.30, 0.70] | 0.60 | 0.35 | 0.42 | 1.00 | 14.2 s (20) |
| naive | 0.50 [0.30, 0.70] | 0.55 | 0.30 | 0.43 | 1.00 | 27.8 s (10) |
| local | 0.65 [0.45, 0.85] | 0.55 | 0.25 | 0.39 | 0.90 | 115.7 s (20) |
| hybrid | 0.60 [0.40, 0.80] | 0.60 | 0.35 | 0.48 | 0.98 | n/a |
| mix | 0.55 [0.35, 0.80] | 0.60 | 0.30 | 0.43 | 1.00 | n/a |
| R1-rules router | 0.60 [0.40, 0.80] | 0.55 | 0.35 | 0.45 | 1.00 | n/a |

¹ Normalised gold answer string contained in the response (judge-free, misses paraphrases).
² Over queries with no cached LLM call (count in brackets); `n/a`: the variant reused keyword or
generation calls cached by an earlier variant.

Paired comparison with V4 (exact McNemar, Holm-corrected): **no variant differs
significantly** (largest: local +0.15, CI [−0.06, 0.36], p_holm = 1.0). R1 routed 9 questions to
vector, 7 to local, 4 to mix.

## 3B → 7B (same 20 questions)

| | 3B run | 7B run |
|---|---|---|
| judge-accepted answers lacking the gold answer (per variant) | 3–6 (5 of 6 inspected for mix plainly wrong) | 0–2 (4 total; 3 are "Jorge Amado" for *Jorge Leal Amado de Faria*, i.e. correct; 1 wrong: "Attack the Block" for *Ant-Man*) |
| gold answer in response (per variant, of 20) | 6–9 | 11–12 |
| EM / F1 | 0.05–0.15 / 0.10–0.21 | 0.25–0.35 / 0.39–0.48 |
| mix judge accuracy | 0.70 (inflated) | 0.55 |
| graph entities / relations | 1,026 / 355 | 1,746 / 841 |
| entities without any relation | 60 % | 48 % |
| malformed relation lines dropped by LightRAG | 1,263 | 52 |
| answer-entity coverage, dev (G3 gate) / all 20 | 4/4 / 14 of 18 | 4/4 / 13 of 18 |

The 3B judge's apparent "mix wins" signal disappears with a better judge: it was judge noise, as
the 3B write-up suspected.

## Findings

1. **An 8B judge is far more consistent with a judge-free check than a 3B judge**, but it still
   has to pass the human agreement gate (κ ≥ 0.6) before its numbers count. The sample for the
   two annotators is exported: `judge_sample.csv` (100 items, verdicts hidden).
2. **Qwen2.5-7B writes 54 % of entity names without spaces** ("HansenClarke", "DemocraticParty",
   "OneWorldTradeCenter"; 951 of 1,746). Two consequences, both fixed in code during this run and
   regression-tested:
   * entity resolution matched normalised names exactly, so 84 nodes in this graph are split
     duplicates ("UnitedKingdom" / "United Kingdom", "JorgeAmado" / "Jorge Amado"). Resolution
     now uses a spacing-insensitive key; **this graph was built before the fix and still contains
     those duplicates** (duplicate rate 4.8 %). Rebuilding merges them.
   * the G3 coverage matcher missed "JaromeIginla" and "SevenDaysBattles"; the first
     validation reported 2/4 and a failed gate. Re-validated with the fixed matcher:
     4/4. `graph_validation.json` holds the re-validated numbers.
3. **Graph modes cost much more time with a local 7B model**: local p50 116 s against 14 s for
   V4, from ~4× larger prompts (graph context) and the keyword-extraction call.
4. Keyword extraction still occasionally returns nothing (8 warnings), so some graph-mode queries
   run with little graph context.
5. Retrieval is not the bottleneck on this subset (recall@10 0.90–1.00 with 10 candidate
   paragraphs per question). Telling retrieval modes apart needs the full datasets (E2), where
   corpora are larger and MuSiQue-style questions defeat lexical retrieval (model-free BM25
   recall@10 on MuSiQue: 0.53).

## Artefacts

* `report/`: tables, paired tests and figures from `elevate-rag report`.
* `graph_validation.json`: G3 coverage and graph shape (re-validated after the matcher fix).
* `triple_audit_sample.csv`: 100 triples with evidence for the human triple audit
  (**not yet annotated**).
* `judge_sample.csv`: 100 judged answers for the human judge check (**not yet annotated**).
* Raw predictions: `runs/G1-7B/` (git-ignored; reproducible from the commands above).
