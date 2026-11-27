# Implementation Plan and Acceptance Gates

The phases follow spec §22 and the plan's gates G1–G4. "Software" criteria are what this
implementation can verify; "Research" criteria need credentials, services, budget or people.

| Phase | Scope | Software acceptance (status) | Research gate |
|---|---|---|---|
| 0 Planning & environment | requirements, decisions, uv env, lockfile, verified APIs/IDs | docs present; `uv sync --locked` works; LightRAG API verified (**done**) | – |
| 1 Data & evaluation foundation | 6 loaders, schemas, splits, manifests, harness, metrics | real loaders validated on real files; end-to-end on synthetic set (**done**) | **G1**: harness end-to-end on 20 real questions with real models: **passed for the pipeline** with $0 local models (`results/g1`, `results/g1_7b`); judge validation is automated (GroUSE + two-judge κ, D-36); no committed run validated yet |
| 2 Vector RAG | V0–V4, chunking ablations, HNSW params | all modes tested; BM25 measured on real data (**done**) | **G2**: tune V4 on dev and freeze (needs models) |
| 3 Graph RAG | LightRAG integration, provenance, resolution, modes, quality checks, community ablation | integration tests vs real LightRAG 1.5.7 (**done**) | **G3**: coverage ≥ 60 % per dataset with the reference extractor (triple audit removed, D-36) |
| 4 Routing | R0–R4, both label schemes, oracle | unit/e2e tests; R1/R2 measured on real labels (**done**) | router results on test after dev-only training |
| 5 DB & model comparisons | adapters, compose, Level A/B, model providers | exact/FAISS/NetworkX verified; server adapters conformance-tested only when services run (**partially verified**) | E4 Level A/B on live services |
| 6 Full evaluation | E1–E6, A1–A4, statistics | runner, estimates, dry runs, stats tested (**done**) | **G4**: runs complete and frozen (needs budget approval) |
| 7 Final QA | audit, regression, docs | ruff, mypy, pytest, coverage, docs (**done**) | – |

## Order of operations for the team

1. Fill `.env` with keys/endpoints, choose the judge model (different family from every generator),
   freeze model IDs in `configs/pipelines/reference.yaml`.
2. `make db-up`; `elevate-rag data prepare all`.
3. `elevate-rag index pilot` per dataset → put the extrapolated cost in the shared tracker → agree
   a hard cap → set `budget.max_usd`.
4. Build indexes, run `index validate` (G3); `judge calibrate` for both judges.
5. Run E1 on dev to tune V4 (chunking, `final_k`, budget), freeze (G2), then E1/E2 on test.
6. Run `E3_labels_dev`, train R2 (both schemes), tune R4 on dev, run E3 on test.
7. E4 (Level A via `bench`, Level B via spec), E5, E6, ablations.
8. Check every run record's `judge_validation` (both judges calibrated, κ ≥ 0.6); cite EM/F1 where unvalidated.
9. `elevate-rag report`; freeze results (G4).
