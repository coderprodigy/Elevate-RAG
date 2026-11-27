# Pending work after the Atlas checkpoint

This backlog was frozen during checkpoint preparation. Priority P0 covers prerequisites for credible research runs, P1 covers required verification and measurements, and P2 covers supporting work. A local item can be implemented or verified in Atlas using suitable fixtures. Full-corpus or paid execution requires the listed dependencies.

## A. Local implementation tasks

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Harden passage index recovery after remote migration or corruption. | P0. Routine reload checks the contract and count. Explicit validation checks full membership. | Isolated remote stores and fixtures containing known IDs. | A same-count index with changed IDs fails closed, recovery remains deterministic, and vector and graph retrieval share the corrected index. | Yes | [Storage](../src/elevate_rag/storage.py), [audit](IMPLEMENTATION_AUDIT.md) |
| Define an explicitly audited policy for cap changes and interrupted runs. | P1. Existing manifests require a new run ID when the cap changes. | Approved recovery semantics and partial-run fixture. | Resume or replacement runs preserve completed evidence, avoid repeated spend and record the old and new manifest and approvals. | Yes with mock spend | [Experiment runner](../src/elevate_rag/experiments.py), [ledger](../src/elevate_rag/providers.py) |
| Extend endpoint-level cost and usage instrumentation for native frameworks. | P1. Native fixtures run, with third-party spend enforcement outside the runner. | Compatible service or mock transport and role attribution contract. | Every request is attributed and bounded per framework, ambiguous failures remain conservatively reserved, and a zero-cap run stays offline. | Yes with mock transport | [Reference worker](../src/elevate_rag/reference_worker.py), [execution plan](RESEARCH_EXECUTION_PLAN.md) |
| Preserve explicit evidence scope and scoring denominators for sources without literal spans. | P1. Published evidence strings are retained and per-span metrics use the mapped subset. | Pinned evaluator interface and representative evidence fixtures. | Reports distinguish missing literal mappings and semantic scores without changing the oracle or held-out inputs. | Partial. Full semantic evaluation needs an authorized model. | [Evaluation](../src/elevate_rag/evaluation.py), [interop](../src/elevate_rag/interop.py) |

## B. Local verification tasks

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Repeat the Phase 2 Qdrant integration check. | P0. Earlier run passed. The rerun was cancelled while Docker was unresponsive. | Responsive local daemon and dedicated Atlas Qdrant service. | Current source passes the opt-in test, reopens the persisted index and validates IDs and HNSW configuration. | Yes when Docker is responsive | [Integration test](../tests/integration/test_live_qdrant.py), [index audit](IMPLEMENTATION_AUDIT.md#physical-passage-index) |
| Recheck live storage adapters, migration and failure handling. | P1. Earlier fixture checks passed. | Dedicated Atlas services, controlled temporary volume or namespace. | Neo4j, Memgraph, Milvus and pgvector regressions and recovery tests pass on current source, with run manifests retained. | Yes with local services | [Live storage tests](../tests/integration/test_live_postgres.py), [Compose](../infrastructure/docker/compose.yaml) |
| Verify live Langfuse and the exact model-serving chat token count. | P1. In-memory tracing and tokenizer unit tests passed. | Dedicated Langfuse deployment and exact model endpoint. | Traces redact prompt and error contents. Fixed messages and provider counts agree on the context budget and generated-token reservation. | Partial | [Tracing tests](../tests/integration/test_reference_tracing.py), [pipeline](../src/elevate_rag/pipeline.py) |
| Reproduce the four framework smoke runs and the seven dataset rebuilds in an independent environment. | P1. Verified on the Atlas Mac. | Locked dependencies, pinned inputs and licensed data access. | A clean host verifies normalized hashes and completes the four fixture runs with isolated model endpoints. | Yes given compute and disk | [Reproducibility](REPRODUCIBILITY.md), [evidence](evidence/README.md) |

## C. Performance and scalability tasks

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Profile corpus and graph growth, embedding batches, cache effectiveness and memory. | P1. Fixed scorer workload measured. Infrastructure-level performance remains open. | Representative development data, frozen embeddings, instrumented server. | Record repeatable p50/p95/p99 latency, QPS, peak RSS, disk and exact recall, with resource limits and corpus hashes. | Partial. Full scale needs study hardware. | [Performance report](PERFORMANCE_REPORT.md), [benchmark code](../src/elevate_rag/benchmarks.py) |
| Profile checkpoint throughput, remote batch checks and artifact growth. | P2. Recovery tested at fixture scale. | Simulated large checkpoint and dedicated service fixture. | A measured bottleneck has a before-and-after profile with identical results and successful resume. | Yes | [Experiment runner](../src/elevate_rag/experiments.py), [storage](../src/elevate_rag/storage.py) |

## D. Deployment portability tasks

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Validate Linux clean setup and service isolation. | P1. macOS ARM test passed. Linux deployment is unverified. | Linux runner and Docker images available for its architecture. | Locked clean install, synthetic checks, explicit service limits, loopback ports and independent data paths verified. | Partial. Needs a Linux host or VM. | [CI](../.github/workflows/quality.yml), [Compose](../infrastructure/docker/compose.yaml) |
| Prepare a future CUDA/EC2 deployment checklist and verify GPU requirements. | P1. No infrastructure provisioned. | Approved GPU specification, region, account and data-handling constraints. | Document VRAM, drivers, CUDA and model versions, budgets, existing resource limits, storage persistence and teardown. Execute provisioning only after explicit authorization. | Planning only | [Model prerequisites](RESEARCH_EXECUTION_PLAN.md#reference-roles-rates-and-preflight), [reference config](../configs/reference.toml) |

## E. External blockers

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Provision and freeze model services and rates. | P0. Public revisions checked. Reference serving and remaining API tiers are open. | Authorized compute for Qwen and Gemma, exact embedding weights, approved API keys and budget. | Freeze revisions, dtype, tokenization and limits. Verify the API reranker rate and the strong API generator checkpoint and exact accounting. | Partial. Candidate configs already exist. | [Reference roles](RESEARCH_EXECUTION_PLAN.md#reference-roles-rates-and-preflight), [suite](../configs/suite.example.toml) |
| Arrange independent annotation and approved research storage. | P0. Reviewers and independent backup destination are unassigned. | Two reviewers, approved reference outcomes and a university-authorized repository or storage system. | Record blinded labels for 50 triples and 100 answers, kappa, any development-only prompt changes and restore-tested backups of the allowed raw artifacts. | No | [Annotation workflow](RESEARCH_EXECUTION_PLAN.md#execution-order-and-gates), [artifact handling](PRE_PUSH_CHECKLIST.md#evidence-preservation) |
| Clarify availability of the optional standalone UltraDomain input. | P2. Data contract exists. | Separately approved licensed input and configuration. | Manifest and data checks pass before the optional run enters the plan. | Partial | [Datasets](DATASETS.md), [known issues](KNOWN_ISSUES.md) |

## F. Full research execution

| Task | Priority and status | Dependencies | Acceptance criteria | Local | References |
| --- | --- | --- | --- | --- | --- |
| Run the E1 and E2 reference experiments and the approved graph pilot. | P0. Matrix and fixture checks complete. Research runs pending. | Frozen models, isolated DBs, approvals, graph review. | V4 is tuned and frozen on development. The 5% pilot and graph quality gate are recorded before held-out evaluation. Native framework outcomes stay isolated. | Partial | [Gates and run order](RESEARCH_EXECUTION_PLAN.md#execution-order-and-gates) |
| Run E3, train outcome-dependent routers and report oracle regret. | P1. Query-type classifiers and grouped CV exist. | Aligned judged E1/E2 development and held-out outcomes. | Outcome labels use development only. Independent classifier data remains isolated. Report aligned accuracy, cost, regret and fallback behavior. | Partial | [Routing](../src/elevate_rag/routing.py), [traceability](REQUIREMENTS_TRACEABILITY.md) |
| Run E4 Level A/B, E5/E6 and approved ablations. | P1. Dry-run matrix exists. | Validated backends, complete model tiers, representative hardware and spend approval. | Match exact FAISS recall, run the 1,000-entity and fixed 300-question protocols, document repeats and compare frozen model factors without changing other settings. | Partial | [Research plan](RESEARCH_EXECUTION_PLAN.md), [benchmark code](../src/elevate_rag/benchmarks.py) |
| Freeze research outcomes and publish supported conclusions. | P1. Report template complete. | Completed paired runs, semantic evaluation, human agreement and fully backed-up artifacts. | Freeze study manifests and results, apply paired statistics and Holm correction, report unknowns and failures, and state only supported research conclusions. | Analysis yes after results exist | [Statistics](../src/elevate_rag/statistics.py), [report template](RESEARCH_REPORT_TEMPLATE.md) |
