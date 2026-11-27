# Phase 2 local delivery

Raw evidence paths below refer to local workspace files. See the [checkpoint evidence](evidence/README.md) for the published summaries.

The implementation audit, fixes and verification are in the uncommitted Atlas worktree. They cover dataset integrity, retrieval scoring, graph quality, context budgeting, embedding and cache handling, reservation accounting, router provenance and native recovery.

- All seven required benchmark datasets passed raw checksum checks and byte-identical normalization. The audit corrected 301 MultiHop-RAG refusal labels and removed within-dataset split overlap.
- The locked clean environment completed 20 of 20 fixture questions with zero errors. The shared-index check reopened and inspected every fixture passage vector. R4 completed and resumed the same 20-question manifest.
- The default suite gained 15 test cases. It finished with 55 passed and three opt-in database cases skipped. Core line coverage is 87.13% and package-wide line coverage is 75.21%. Ruff lint, formatting, mypy and the uv lock check passed.
- All four native comparison frameworks have local fixture evidence. This phase added Microsoft GraphRAG 3.3.0 and HippoRAG 2.0.0a5, both with two of two questions completed using loopback fixture models.
- The measured scoring stage went from 494.8 ms to 11.6 ms for the fixed 100-question workload. Both scoring paths produced the same result fingerprint.
- All six experiment blocks have planning artifacts. E5 has 54 configured arms and E6 has 24. Their remaining arms need an approved API reranker rate or a frozen strong API generation configuration. Reference experiments, paid scoring, human annotation and study conclusions remain open.

The remote storage checks were captured during the implementation pass. A fresh Phase 2 Qdrant repeat was cancelled after Docker stopped responding. Phase 2 FAISS persistence and full vector membership were verified locally.

[Defects and fixes](IMPLEMENTATION_AUDIT.md), [requirement classifications](REQUIREMENTS_TRACEABILITY.md), test output (local evidence, `artifacts/phase2/tests.txt`), [measured performance](PERFORMANCE_REPORT.md), [exact reproduction commands](REPRODUCIBILITY.md), [open issues](KNOWN_ISSUES.md), [research execution plan](RESEARCH_EXECUTION_PLAN.md)

## Git state

`atlas` remains at `7c27716`. The implementation and audit files are untracked, and the staging area is empty. Local benchmark data, reports and caches are ignored. There were no commits, branch changes, pushes or paid model calls.
