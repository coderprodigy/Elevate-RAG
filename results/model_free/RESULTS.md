# Model-free measurements on real data (PARTIAL RESULTS)

Produced by `uv run python scripts/model_free_measurements.py` on 2026-11-26 (raw numbers with
dataset versions and configuration: `model_free_measurements.json`). They need no LLM,
embedder or credentials. **They are not end-to-end QA results:** no generation, no judge.
All intervals are 95 % bootstrap CIs over test-split questions (1,000 resamples).

Configuration: reference chunking (512 tokens, 64 overlap, sentence-aware, title prepended),
BM25 k1 = 0.9, b = 0.4, document-level recall (a gold document counts if any of its chunks is
retrieved).

## V1 (BM25) retrieval, test split, answerable questions

| Dataset | n | Recall@10 | Recall@50 | nDCG@10 | Evidence recall@10 |
|---|---:|---|---|---|---|
| NQ (gold + 4.3k distractor passages) | 400 | 0.782 [0.743, 0.823] | 0.845 [0.807, 0.877] | 0.674 [0.635, 0.716] | n/a (no evidence text) |
| HotpotQA (distractor, hard) | 800 | 0.821 [0.803, 0.838] | 0.932 [0.920, 0.943] | 0.730 [0.712, 0.746] | n/a |
| MuSiQue (answerable) | 400 | 0.525 [0.497, 0.554] | 0.682 [0.655, 0.710] | 0.507 [0.483, 0.532] | n/a |
| MultiHop-RAG | 1,804 | 0.842 [0.831, 0.853] | 0.951 [0.945, 0.957] | 0.735 [0.724, 0.746] | 0.657 [0.643, 0.670] |
| GraphRAG-Bench medical | 1,649 | undefined | undefined | undefined | undefined* |
| GraphRAG-Bench novel | 1,609 | undefined | undefined | undefined | undefined* |

\* GraphRAG-Bench has a single document per corpus (no gold document IDs), and its gold evidence
is paraphrased: only 0.45 % (medical) and 0.75 % (novel) of evidence strings occur verbatim in
the corpus, against 100 % for MultiHop-RAG. String-match evidence recall would report a
meaningless near-zero value, so it is withheld. These datasets need the benchmark's LLM-judged
evidence recall.

Reading: MuSiQue, built to block shortcuts, is the hardest for lexical retrieval (recall@10
0.53), consistent with the plan's expectation that it favours graph retrieval. The QA-GraphRAG
point that BM25 is a strong baseline holds on HotpotQA and MultiHop-RAG.

## R1 rule router: query-type accuracy against dataset labels (test split)

| Dataset | Accuracy |
|---|---|
| NQ | 0.968 [0.950, 0.985] |
| HotpotQA | 0.453 [0.419, 0.488] |
| MuSiQue | 0.542 [0.500, 0.590] |
| MultiHop-RAG | 0.454 [0.433, 0.474] |
| GraphRAG-Bench medical | 0.523 [0.500, 0.546] |
| GraphRAG-Bench novel | 0.320 [0.296, 0.343] |

The rules were written before these numbers were computed and were **not** tuned on them.

## R2 TF-IDF + linear SVM on RAGRouter-Bench (7,727 queries, 3 labels)

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| 5-fold stratified CV | 0.936 | 0.931 |
| Leave-one-domain-out (mean) | 0.659 | 0.671 |
| held out: GraphRAG-Bench medical | 0.466 | 0.457 |
| held out: MuSiQue | 0.634 | 0.658 |
| held out: QuALITY | 0.787 | 0.796 |
| held out: UltraDomain legal | 0.890 | 0.885 |

The 5-fold result matches the 93.2 % / 0.928 reported by Bansal & Agarwal (2026), though the
protocol details may differ. Accuracy falls by about 28 points when a whole domain is held out.
That is the plan's risk "router overfits to one dataset's phrasing", now measured, and it supports
reporting R2 with leave-one-dataset-out alongside in-domain CV.
