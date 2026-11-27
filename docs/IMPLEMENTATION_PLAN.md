# Implementation plan

| Phase | Dependency | Check |
| --- | --- | --- |
| 0. Design | Read specification and full proposal, inspect OS and dependency cache | Document decisions, interfaces and requirement mapping |
| 1. Data and evaluation | Normalized document/question contracts and stable identities | Run 20 deterministic synthetic questions through persisted evaluation and reports |
| 2. Vector | Shared corpus, tokenizer and providers | Verify V0–V4 and freeze only after development tuning on real data |
| 3. Graph | Shared corpus, revisions and storage contracts | Smoke-test actual LightRAG retrieval and validate passage IDs. Measure graph and prepare audit |
| 4. Routing | Static pipeline outcomes and offline development labels | Verify R0–R4, cross-validation, held-out protection and oracle calculations |
| 5. Experiment tooling | Indexes, stages and provider metrics | Validate experiment matrix, storage services and reproducible run identity |
| 6. Research evaluation | Approved budget, dataset versions, pinned models and human audit | Run authorized blocks, collect paired outcomes and freeze results |
| 7. Delivery | All feasible verification | Verify clean installation, CLI, lint, types, tests and traceability |
