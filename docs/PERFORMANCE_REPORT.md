# Phase 2 performance report

Raw evidence paths below refer to local workspace files. See the [checkpoint evidence](evidence/README.md) for the published summaries.

## Measured changes

The profile used Python 3.12.13 on the Apple M4 host with 16 GB RAM. The workload scored 100 published Natural Questions development questions against a catalog of 18,647 passages. The candidate list was fixed and the timed calls excluded file loading. Both paths used the same evidence and candidate ordering.

| Workload | Before | After | Result |
| --- | ---: | ---: | --- |
| Passage ranking metrics, median of five runs | 494.8 ms | 11.6 ms | 42.5× faster, including construction of the per-document lookup in every measurement. Both paths produced the same score SHA-256. |
| Duplicate embedding contract fixture, input `[a, b, a]` | 3 model rows | 2 model rows | All three output rows keep their original order, and the cached follow-up issues no second request. |
| Clean, locked offline V4 fixture | — | 20 of 20 completed | The offline report records an 11.4 ms fixture mean per question. |
| Shared graph/vector fixture validation | — | 20 of 20 indexed vectors inspected | The validation checked every configured passage ID after reopening the index. |

The pre-change profiler found an ideal nDCG denominator scan over the catalog for every question. The production scorer and V4 tuner build a per-document lookup once per run or tuning arm. That confines the calculation to documents with evidence while still checking exact span overlap. The benchmark compares the same scoring implementation with and without that lookup to isolate the optimization.

The profiles quantify evaluation overhead. Provider throughput and relative database performance remain outside this profile. The embedding fixture measures submitted rows. Actual token and latency savings depend on corpus duplication, batch boundaries and cache history. Controlled E4 comparisons still require frozen real embeddings, a production-shaped graph, identical resource budgets and the specified repetitions.

Profile and scoring fingerprints (local evidence, `artifacts/phase2/performance/metrics.json`), original profile timing (local evidence, `artifacts/phase2/before-metrics.json`), [regression tests](../tests/integration/test_phase2_regressions.py), clean run (local evidence, `artifacts/phase2/clean-results/synthetic/phase2-clean-v4/report.md`)

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  scripts/profile_retrieval_metrics.py data/natural-questions artifacts/profiling
uv run --no-sync python -m pstats artifacts/profiling/full_scan.prof
```
