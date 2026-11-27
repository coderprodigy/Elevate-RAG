# Known issues and open research gates

| Status | Scope | Dependency or limitation |
| --- | --- | --- |
| Externally blocked | Controlled E1–E6 and G2–G4 | The candidate revisions exist publicly. The local Qwen endpoint is absent, the pinned model files need provisioning, and approved paid budget and credentials are required. Two reviewers are needed for the graph and judge audits. |
| Open | E4 research measurements | Each specified store passed a fixture check. The controlled fixed-query repeats, 1,000-entity sample, matched-recall operating points and service-level resource collection await study execution. |
| Open | E5/E6 model tiers | The candidate suite has four embedding tiers, three extraction tiers, two reranker settings and two generation tiers. The API reranker rate and the strong API generator's frozen endpoint and exact token accounting remain open. |
| Open | Independent metrics and judge calibration | Official GraphRAG-Bench schema and RAGAS transport have been checked. The real semantic scores, reference judge, paired global outcomes and human agreement remain pending. |
| Open | Instrumentation | Langfuse has an in-memory SDK integration check. A live loopback Langfuse deployment needs its own verification. Third-party frameworks require separate usage and spend enforcement at model endpoints. |
| Limitation | Exact evidence spans | The Xiang datasets and RAGRouter-Bench preserve the published evidence strings. Per-span metrics use the mapped subset and the upstream semantic methods require an authorized evaluator run. |
| Limitation | Cross-benchmark classification | RAGRouter-Bench overlaps MuSiQue and GraphRAG-Bench medical. The grouped CV is an independent study. Real R2 evaluation requires the current development partition of the active dataset. The Natural Questions query-type development partition has one class, so its R2 query-type arm reports a constant prediction. |
| Limitation | Remote index rehydration | Routine reuse checks the stored contract and collection size. Run `validate-index` to check full membership after a service migration or suspected corruption. Services must use the same isolated endpoint and namespace for continued physical reuse. |
| Limitation | Recovery across an altered configuration | A changed immutable manifest, including a changed cap in the configuration, requires a distinct run ID. With the same manifest and ledger, `--retry-failed` retries failed checkpoints. |
| Externally blocked | Optional UltraDomain | The workflow accepts separately prepared inputs. No separate UltraDomain dataset was supplied for this audit. |

[Phase 2 audit](IMPLEMENTATION_AUDIT.md), [research prerequisites](RESEARCH_EXECUTION_PLAN.md)
