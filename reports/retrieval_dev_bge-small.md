# Retrieval on dev splits: V1–V4 with bge-small (2026-11-25)

Generated with `hgrag index evaluate --embedder bge-small`. This is retrieval only, with no LLM calls.

- **Setup:** bge-small is the small dev embedder; the reference embedder (Qwen3-Embedding-0.6B) is still being built. Reranker: bge-reranker-v2-m3 (`rerank_candidates` = 50). final_k = 8. Budget: 8k tokens.
- **What the columns measure:** retrieval at chunk level. "R@k" is the share of gold chunks in the top k. "hit@5" means at least one gold chunk is in the top 5 (the standard NQ metric).
- **R@10 equals R@8**, because only `final_k` = 8 passages are returned.
- **GraphRAG-Bench:** gold chunks come only from string-matched evidence (about 2% of its questions; n = 20 and 33). Its main retrieval metric will be LLM-checked evidence recall.

| Dataset | Variant | R@1 | R@5 | R@10 | hit@5 | nDCG@10 | doc R@5 | ~ctx tokens | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nq | v1 BM25 | 0.071 | 0.218 | 0.322 | 0.580 | 0.343 | 0.218 | 1,080 | 100 |
| nq | v2 dense | 0.231 | 0.601 | 0.685 | 0.910 | 0.702 | 0.601 | 1,084 | 100 |
| nq | v3 hybrid | 0.128 | 0.473 | 0.612 | 0.850 | 0.560 | 0.473 | 1,080 | 100 |
| nq | **v4 hybrid+rerank** | 0.318 | **0.627** | 0.692 | **0.940** | **0.752** | 0.627 | 1,084 | 100 |
| hotpotqa | v1 BM25 | 0.385 | 0.728 | 0.810 | 0.965 | 0.729 | 0.728 | 896 | 200 |
| hotpotqa | v2 dense | 0.453 | 0.848 | 0.890 | 0.995 | 0.851 | 0.848 | 963 | 200 |
| hotpotqa | v3 hybrid | 0.432 | 0.833 | 0.895 | 1.000 | 0.830 | 0.833 | 940 | 200 |
| hotpotqa | **v4 hybrid+rerank** | 0.477 | **0.940** | 0.955 | 1.000 | **0.923** | 0.940 | 985 | 200 |
| musique | v1 BM25 | 0.273 | 0.418 | 0.470 | 0.820 | 0.482 | 0.419 | 898 | 100 |
| musique | v2 dense | 0.296 | 0.542 | 0.595 | 0.930 | 0.587 | 0.543 | 1,026 | 100 |
| musique | v3 hybrid | 0.316 | 0.547 | 0.593 | 0.960 | 0.598 | 0.547 | 971 | 100 |
| musique | **v4 hybrid+rerank** | 0.342 | **0.623** | 0.672 | 0.960 | **0.677** | 0.624 | 1,112 | 100 |
| multihop_rag | v1 BM25 | 0.184 | 0.487 | 0.581 | 0.849 | 0.494 | 0.669 | 3,807 | 451 |
| multihop_rag | v2 dense | 0.123 | 0.351 | 0.438 | 0.701 | 0.361 | 0.549 | 3,624 | 451 |
| multihop_rag | v3 hybrid | 0.173 | 0.442 | 0.544 | 0.794 | 0.461 | 0.628 | 3,786 | 451 |
| multihop_rag | v4 hybrid+rerank (50-question sample) | 0.233 | 0.543 | 0.641 | 0.902 | 0.586 | 0.705 | 3,774 | 41 |
| graphrag_bench_novel | v1 BM25 | 0.450 | 0.775 | 0.825 | 0.800 | 0.669 | 1.000 | 3,895 | 20 |
| graphrag_bench_novel | v2 dense | 0.275 | 0.650 | 0.750 | 0.650 | 0.508 | 0.950 | 3,852 | 20 |
| graphrag_bench_novel | v3 hybrid | 0.475 | 0.825 | 0.825 | 0.850 | 0.690 | 0.950 | 3,899 | 20 |
| graphrag_bench_novel | **v4 hybrid+rerank** | 0.600 | 0.825 | 0.825 | 0.850 | **0.732** | 1.000 | 3,911 | 20 |
| graphrag_bench_medical | v1 BM25 | 0.288 | 0.606 | 0.667 | 0.606 | 0.483 | 1.000 | 3,991 | 33 |
| graphrag_bench_medical | v2 dense | 0.298 | 0.586 | 0.677 | 0.636 | 0.501 | 1.000 | 3,984 | 33 |
| graphrag_bench_medical | v3 hybrid | 0.333 | 0.662 | 0.752 | 0.697 | 0.547 | 1.000 | 3,983 | 33 |
| graphrag_bench_medical | **v4 hybrid+rerank** | 0.450 | **0.778** | 0.808 | 0.788 | **0.652** | 1.000 | 3,988 | 33 |

## Early observations (dev only; not final results)

1. **V4 (hybrid + reranker) is the best retriever everywhere.** This supports the plan's choice of V4 as the reference baseline.
2. **BM25 beats dense on the long-chunk news and novel corpora** (MultiHop-RAG, GraphRAG-Bench novel) when bge-small is the embedder. This is consistent with QA-GraphRAG's finding that lexical retrieval is a strong baseline.
3. **Plain RRF hybrid (V3) can be worse than the better single retriever.** Examples: NQ, where dense is far stronger than BM25, and MultiHop-RAG, where the reverse holds. Equal-weight fusion pulls toward the weaker side. The reranker recovers this.
4. **Gold retrieval is far from saturated** on MuSiQue (R@5 0.62) and MultiHop-RAG (about 0.54). These multi-hop sets are where graph retrieval should have room to help (RQ2).
