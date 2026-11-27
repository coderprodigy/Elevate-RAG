# Research execution plan

Raw evidence paths below refer to local workspace files. See the [checkpoint evidence](evidence/README.md) for the published summaries.

The study remains blocked on frozen model infrastructure, paid provider approval and human annotation. The offline configuration is suitable for software validation. The real configuration retains a zero-dollar cap until the study budget is approved.

## Reference roles, rates and preflight

| Role | Reference candidate | Confirmed and pending |
| --- | --- | --- |
| Embedding and chunk tokenizer | BAAI/bge-m3, 1,024 dimensions | Exact Hugging Face revision returned HTTP 200. Provision the weights and tokenizer locally. |
| Cross-encoder | BAAI/bge-reranker-v2-m3 | Exact revision returned HTTP 200. Provision weights locally. |
| Generator and chat tokenizer | Qwen3.6-35B-A3B at the candidate revision | The public revision resolved, and no service listened at the configured loopback port. Freeze model weights, dtype or quantization, serving version, chat template and context behavior. |
| Extraction, keyword and R3 routing | Claude Haiku 5.5 | The fixed model ID and compatibility endpoint are documented. Configure workspace-scoped API keys when authorized. The candidate reserves the upper advertised prompt-length tier. |
| Independent judge | DeepSeek V4 Pro, `DeepSeek-V4-Pro-0813` | Published version and API route checked. The estimate uses peak cache-miss input at $1.32 and peak output at $3.96 per million tokens. The host has no authorized paid test result. |

[Candidate config](../configs/reference.toml), public revision and local listener checks (local evidence, `artifacts/phase2/model-availability.json`), [Anthropic models and pricing](https://platform.claude.com/docs/en/about-claude/pricing), [Anthropic OpenAI compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk), [DeepSeek pricing and limits](https://api-docs.deepseek.com/quick_start/pricing/)

The suite includes candidate definitions for all four embedding tiers, three extractors, two reranker settings and two generation tiers. It includes the exact public BGE-small, Qwen3-Embedding-8B and Gemma 4 12B revisions, plus the OpenAI embedding API candidate. Confirm BGE-small's 512-token limit and any model-specific truncation on development data while preserving the common passages. The API reranker needs an approved rate card. The strong API generator needs a fixed serving revision and a compatible exact chat-token measurement. Record data parallelism, device, database versions, index parameters and memory limits. Validate the passage dimension and tokenizer with one local request before any authorized paid batch. Store credentials in environment variables. Use a dedicated study artifact directory for the SQLite budget ledger. [Tier config](../configs/suite.example.toml), public tier revisions (local evidence, `artifacts/phase2/tier-models.json`), [OpenAI embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-large), [Jina API contract](https://jina.ai/reranker)

## Size, timing and spend assumptions

| Block | Planned internal evaluation rows | Additional scope |
| --- | ---: | --- |
| E1 | 34,915 | 30 V0–V4 arms |
| E2 | 34,915 | 14,612 external-framework question rows across eight arms |
| E3 | 76,813 | 6,983 R3 classifications, up to 6,983 extra R4 generations, CV and offline oracle |
| E4 | 14,400 | 48,000 warm-up queries and eight Level A measurements |
| E5 | 80,915 | 66 model-role arms |
| E6 | 40,842 | 36 mode × generator arms |

These are planned rows derived from the current frozen dataset counts and code matrix. Shared static results may reduce actual generation when exact fingerprints and split provenance match. The envelope assumes 337,783 internal generation calls including warm-ups and the maximum one-fallback allowance. At 2,800 input and 200 output tokens per call, this gives about 946 million input and 67.6 million output tokens. At a hypothetical serial 2–6 seconds per generation, it gives 188–563 generator-hours. These are throughput scenarios before indexing, judge time, external runs, retries or concurrency effects.

A one-grade-per-internal-row scenario assumes 1,200 input and 200 output tokens per judge call. At the configured peak rates it totals approximately $672. Indexing, RAGAS, global pairwise judging, paid model tiers and transport retries are outside that number. For a 100,000-passage scenario with one Claude extraction request per chunk and 1,200 input plus 250 output tokens, the short-prompt rates imply about $24.50. The conservative upper tier used for reservations produces a higher estimate. Replace these assumptions with reported counts and provider invoices from the approved pilot before judging affordability. Machine-readable estimate (local evidence, `artifacts/phase2/study-estimate.json`), [rate card](https://platform.claude.com/docs/en/about-claude/pricing)

A 1,024-dimensional float32 passage vector needs 4,096 bytes before index structures. A 100,000-passage matrix would be about 391 MiB and a separately persisted ANN representation adds its own storage. Measure graph profiles, HNSW overhead, container files and memory from the pilot. The Qwen model has about 35B total parameters, so bf16 parameters alone need about 70 GB in decimal units. The M4 host has 16 GB unified RAM. Provision a separate reference model host with headroom for the serving stack and KV cache. A lower-precision option requires a new frozen study configuration.

## Execution order and gates

1. Reproduce the data and fixture checks. Verify checksums, split memberships, local model metadata, database services and expected HNSW settings. Keep RAGRouter-Bench independent of main deployment training.
2. Tune V4 on development evidence only. Record the chosen settings and preserve the generated frozen JSON and tuning manifest. Audit V0 behavior, power on paired development scores and cap viability.
3. Build the reproducible 5% passage graph pilot for every paid indexing configuration. Record wall time, per-role token usage, cache state, projected full cost, graph size and machine limits. The budget system reserves capacity atomically before requests. Full paid indexing compares the pilot estimate against all existing ledger reservations. Retain conservative reservations after ambiguous transport failures.
4. Build the chosen full graph after pilot approval. Inspect development answer-entity coverage and review the 50-triple audit. Apply the schema extraction fallback if permitted tuning still leaves the required coverage below 60%. Record the fallback configuration and repeat the audit.
5. Run aligned development V0–V4 and graph mode anchors, and calibrate the judge and R4 threshold before freezing. The six query-type classifiers are prepared locally, with a reported constant V4 class for Natural Questions. Train the outcome classifier on fixed development arms. Freeze every model tier, judge prompt and retrieval configuration, including the matching persisted development SHA and dependency set.
6. Run held-out E1 and E2, then align the already-frozen router, oracle and final comparisons for E3. Keep native framework outputs isolated. Run E4 Level A with exact FAISS, matched recall and 1/8/32-client measurements. Run E4 Level B on the fixed 300-question subset with development-only warm-ups and three repetitions. Complete E5 and E6 from the frozen factor definitions.
7. Resume using the identical run ID, code, configuration, data and output root. SQLite keeps one outcome per question. Pass `--retry-failed` for budget or transport errors after the blocker is resolved. Use a new run ID for changes that affect the manifest. Native checkpoints recover from a partial trailing line. External native model traffic must be explicitly budgeted in its serving layer.
8. Compare aligned outcomes, apply the paired resampling and Holm correction, export plots and freeze all reports before interpretation. Describe the denominator and unavailable data in each metric. Retain all recorded failure counts.

For graph review, two annotators independently use the sampled triple and its actual passage excerpts to assign correct, partial or wrong. For judge calibration, each reviews the same 100 sampled answers with questions and references, without access to model identities, the other reviewer’s labels or the automated judge grades. Keep original labels and any adjudication separate. Use `audit-agreement` to calculate pairwise kappa. Revise the judge prompt on development data if agreement falls below 0.6. Keep ratings out of the held-out generator inputs. [Audit formats and statistical protocol](EXPERIMENTS.md)

```sh
for block in E1 E2 E3 E4 E5 E6; do
  uv run --no-sync elevate-rag --config configs/reference.toml \
    suite "$block" configs/suite.example.toml --split heldout
done
uv run --no-sync python scripts/estimate_study.py
```

The commands above validate the planning matrix and compute estimates. `suite --execute` is for the approved study environment with usable frozen models. The fixture run, metadata probes and dry runs made no paid requests.
