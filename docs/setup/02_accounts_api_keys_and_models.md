# 02 — Accounts, API keys and the model freeze

## 1. Which accounts you need

| Provider | Used for | Required? | Env var |
|---|---|---|---|
| **Anthropic** (console.anthropic.com) | Extraction LLM (Haiku 5.5 reference, Sonnet 5.5 comparison), strong API generator (Sonnet 5.5), R3 router | **Yes** | `ANTHROPIC_API_KEY` |
| **OpenAI** *or* **Google Gemini** | **Judge** (must be a model family not used by any generator), RAGAS LLM, optional API embedder | **Yes** (one of them) | `OPENAI_API_KEY` / `GEMINI_API_KEY` |
| **OpenRouter** (or Together / Fireworks) | Open-weight generators and extractor (Gemma 4 12B, Qwen3.6-35B-A3B or DeepSeek V4-Flash) | Yes, unless you use university GPUs | `OPENROUTER_API_KEY` |
| Cohere or Jina | API reranker comparison (E5) | Should-tier | `COHERE_API_KEY` / `JINA_API_KEY` |
| Voyage | API embedder comparison (E5) | Optional | `VOYAGE_API_KEY` |
| Langfuse Cloud (free Hobby plan) | Tracing (`LANGFUSE_*`) | Optional: our own ledger already tracks cost | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| Hugging Face | Dataset downloads (some datasets need you to accept terms) | Recommended | `HF_TOKEN` |

**Spending caps.** Set a monthly limit in each provider's console as a hard backstop. The project's own
budget guard (`configs/budget.yaml`) stops runs before that limit is reached.

## 2. Create `.env`

Phase 0 creates `.env.example`. Copy it and fill it in:

```bash
cp .env.example .env      # .env is git-ignored; never commit it
```

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
HF_TOKEN=
# optional
COHERE_API_KEY=
VOYAGE_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Each team member uses their own `.env`. If several people share one key, the ledger's `user` column
(taken from `git config user.name`) still separates spend by person.

## 3. Model freeze (protocol: freeze in week 1)

Fill this table as a team. I'll copy the result into `configs/models.yaml` in Phase 0, including prices.
After that, changing a model means a new config version, never an edit in place.

| Role | Tier | Proposed model | Exact ID / provider + quantisation | $/1M in | $/1M out |
|---|---|---|---|---|---|
| Extractor | cheap API (**reference**) | Claude Haiku 5.5 | `claude-haiku-5-5` | 0.10 | 0.50 |
| Extractor | strong API | Claude Sonnet 5.5 | `claude-sonnet-5-5` | 2.00 | 10.00 |
| Extractor | open | Qwen3.6-35B-A3B | _fill in (hosted)_ | | |
| Generator | small open | Gemma 4 12B | _fill in_ | | |
| Generator | mid open (**reference**) | Qwen3.6-35B-A3B or DeepSeek V4-Flash | _fill in_ | | |
| Generator | strong API | Claude Sonnet 5.5 | `claude-sonnet-5-5` | 2.00 | 10.00 |
| Judge | fixed | strongest GPT or Gemini model | _fill in_ | | |
| Router R3 | cheap | Claude Haiku 5.5 | `claude-haiku-5-5` | 0.10 | 0.50 |
| Embedder | small open | bge-small-en-v1.5 | local | 0 | — |
| Embedder | strong open (**reference**) | Qwen3-Embedding-0.6B (8B if compute allows) | local / hosted | | |
| Embedder | API | text-embedding-3-large / Voyage / Gemini | _fill in_ | | |
| Reranker | open (**reference**) | bge-reranker-v2-m3 | local | 0 | — |
| Reranker | API | Cohere or Jina rerank | _fill in_ | | |

The Claude prices are Anthropic's first-party list prices for Haiku 5.5 (prompts up to 100K tokens) and
Sonnet 5.5, checked against the published price list in Nov 2026. Confirm them on the pricing pages on
freeze day.

**Note on determinism.** Claude Haiku 5.5 and Sonnet 5.5 reject non-default `temperature`, and the
Anthropic API has no `seed` parameter. We use temperature 0 where providers allow it, minimal thinking
effort for Claude models, the response cache for reruns, and a measured variance check. This is explained
in the plan, §1.2-1.

## 4. Budget

Agree a **hard total cap** (for example $150–300 for the Must and Should tiers) and per-phase caps. I'll put them
in `configs/budget.yaml`. The largest costs, in order: graph extraction (especially with Sonnet 5.5), the
judge and RAGAS over all runs, and the strong API generator in E5/E6.
