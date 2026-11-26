# Models

Every model Elevate-RAG calls is a configuration value in the Model_Registry,
`configs/models.yaml`. No model identifier is hard-coded, and every LLM and embedding call goes
through the LLM_Gateway using the registry entry for its role. This document lists the registry
entries, explains how to obtain or serve the open-weight models, and how to substitute a model you
cannot use.

`configs/models.yaml` is the source of truth. If it and this document disagree, the YAML wins, and
this document should be updated.

## Contents

1. [Roles and registry fields](#1-roles-and-registry-fields)
2. [Registry entries](#2-registry-entries)
3. [Obtaining and serving local open-weight models](#3-obtaining-and-serving-local-open-weight-models)
4. [Substituting an unavailable model](#4-substituting-an-unavailable-model)
5. [Provider-specific notes](#5-provider-specific-notes)

## 1. Roles and registry fields

| Role | Used by | Reference choice (Requirement 27.2) |
|---|---|---|
| `embedder` | chunk, entity-profile and relation embeddings; dense retrieval | strong open embedder |
| `reranker` | cross-encoder rescoring for V4, mix mode and R4 confidence | open reranker |
| `extraction` | entity and relation extraction, description summaries, community reports | cheap API extractor |
| `generator` | answers from numbered context | mid-size open generator |
| `judge` | correct/partial/wrong grading and pairwise comparisons | API model from a different family than the generator |

Each entry records (Requirement 3.1):

- provider (`fake`, `sentence-transformers`, `openai`, `anthropic`, `google`, `ollama`, `vllm`)
- exact model identifier, plus version or snapshot date where the provider offers one
- model family, used to reject a judge from the same family as the generator (Requirement 3.8)
- context length in tokens
- price in US dollars per million input tokens and per million output tokens

Prices are copied from the provider's pricing page on the day the registry is frozen and are used
by the Budget_Tracker. Local models have a price of 0; their cost shows up as wall-clock time, not
dollars.

## 2. Registry entries

Status meanings: **verified** means the model ID was found on the provider's model page or the
Hugging Face Hub when this document was written. **choose at freeze** means the plan fixes the
role but not the exact model; pick one when you freeze the registry and write it into
`configs/models.yaml`.

### Offline fakes (default)

These need nothing and are what the exit checks and tests use.

| Key | Role | Provider | Notes |
|---|---|---|---|
| `fake-embedder` | embedder | fake | Hash-based deterministic vectors |
| `fake-reranker` | reranker | fake | Deterministic scores |
| `fake-llm` | extraction, generator, judge | fake | Scripted or record-replay responses |

### Embedders

| Key | Model ID | Provider | Context | Size on disk | Licence | Status |
|---|---|---|---|---|---|---|
| `bge-m3` (reference) | `BAAI/bge-m3` | sentence-transformers (local) | 8,192 | about 2.3 GB | MIT | verified |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | sentence-transformers (local) | 32,768 | about 1.2 GB | Apache-2.0 | verified |
| `minilm-l6` | `sentence-transformers/all-MiniLM-L6-v2` | sentence-transformers (local) | 256 | about 90 MB | Apache-2.0 | small CPU-friendly option for laptops |
| `openai-embedding` | choose at freeze (an OpenAI `text-embedding-*` model) | openai | provider limit | n/a | API terms | choose at freeze; needs `OPENAI_API_KEY` |

Changing the embedder means re-embedding the corpus. A query embedded with a different model than
the index stops with an `EmbeddingMismatchError` naming both models.

### Rerankers

| Key | Model ID | Provider | Size on disk | Licence | Status |
|---|---|---|---|---|---|
| `bge-reranker-v2-m3` (reference) | `BAAI/bge-reranker-v2-m3` | sentence-transformers CrossEncoder (local) | about 2.3 GB | Apache-2.0 | verified |
| `ms-marco-minilm-l6` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers CrossEncoder (local) | about 90 MB | Apache-2.0 | small CPU-friendly option |

### Extraction LLMs

| Key | Model ID | Provider | Status |
|---|---|---|---|
| `extractor-api` (reference) | choose at freeze: OpenAI's current low-cost model (see <https://platform.openai.com/docs/models>) | openai | choose at freeze; needs `OPENAI_API_KEY` |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | vllm or ollama (local) | verified; alternative extractor for the extractor comparison |

Extraction is the largest cost in the project. Run the 5% cost estimate before full indexing.
Extractor comparisons are limited to MultiHop-RAG and GraphRAG-Bench medical (Requirement 27.3).

### Generators

The three generators of the E6 factorial (mode × generator):

| Key | Model ID | Provider | Context | Price in / out per MTok | Licence | Status |
|---|---|---|---|---|---|---|
| `qwen3.6-35b-a3b` (reference) | `Qwen/Qwen3.6-35B-A3B` (also `Qwen/Qwen3.6-35B-A3B-FP8`) | vllm or ollama (local) | see model card | 0 (local) | Apache-2.0 | verified on the Hub |
| `gemma-4-12b` | `google/gemma-4-12B-it` (also `google/gemma-4-12B-it-qat-q4_0-gguf` for llama.cpp/Ollama) | vllm or ollama (local), or google API | see model card | 0 (local) | Apache-2.0 | verified on the Hub; not gated when checked |
| `claude-sonnet-5.5` | `claude-sonnet-5-5` | anthropic | 1M | $2 / $10 | API terms | verified on Anthropic's model page |

### Judge

| Key | Model ID | Provider | Status |
|---|---|---|---|
| `judge-api` | choose at freeze: an OpenAI model strong enough to reach Cohen's kappa ≥ 0.6 against human labels | openai | choose at freeze; needs `OPENAI_API_KEY` |

The judge family must differ from the generator family in every Run. An OpenAI judge works with
all three planned generators (Qwen, Gemma, Claude). If you use an OpenAI model as a generator, pick a
judge from another family, or the Config_Loader stops with an error naming both models.

### Decisions you need to make

1. The exact OpenAI model IDs for `extractor-api` and `judge-api`, and their prices on the freeze
   date.
2. Whether to serve Qwen3.6-35B-A3B and Gemma 4 12B locally (GPU needed, see
   [SETUP.md §2](SETUP.md#2-hardware-assumptions)) or through a hosted OpenAI-compatible endpoint.
3. Whether to include the optional API embedder in the embedder comparison.

## 3. Obtaining and serving local open-weight models

Install the `ml` extra for local embedders and rerankers and the `providers` extra for local LLM
servers (both speak the OpenAI API, so the `openai` client is used):

```bash
uv sync --frozen --extra ml --extra providers   # add other extras you need in the same command
```

Weights are cached under `HF_HOME` (default `./.cache/huggingface`). Set `HF_TOKEN` if a
repository is gated or you hit download rate limits; for gated models, also accept the licence on
the model's Hub page with the same account.

### Embedders and rerankers (CPU or GPU)

Download once with network access, then everything runs offline:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-m3')"
uv run python -c "from sentence_transformers import CrossEncoder as C; C('BAAI/bge-reranker-v2-m3')"
# small laptop options
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
uv run python -c "from sentence_transformers import CrossEncoder as C; C('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

`elevate verify-setup` reports each local model in the registry as `pass` when its weights are in
the cache, and `skipped` otherwise. On a laptop CPU, bge-m3 embeds a Tiny_Sample in a few minutes;
full corpora are much faster on a GPU or Apple Silicon (`mps`).

### LLMs with vLLM (Linux + NVIDIA GPU)

vLLM serves an OpenAI-compatible API at `VLLM_BASE_URL` (default `http://localhost:8000/v1`).

```bash
pip install vllm          # in a separate environment; see https://docs.vllm.ai for CUDA versions
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --api-key "$VLLM_API_KEY" --max-model-len 32768
# or
vllm serve google/gemma-4-12B-it   --port 8000 --api-key "$VLLM_API_KEY" --max-model-len 32768
```

Serve one model per port. If you serve two at once, give the second a different port and point
its registry entry at that URL. Check that the model is up with
`curl -H "Authorization: Bearer $VLLM_API_KEY" $VLLM_BASE_URL/models`.

### LLMs with Ollama (macOS, Windows, Linux; CPU or GPU)

Ollama serves an OpenAI-compatible API under `$OLLAMA_BASE_URL/v1`.

```bash
# install from https://ollama.com/download, then:
ollama pull <tag>          # find the tag for Qwen3.6-35B-A3B or Gemma 4 12B at https://ollama.com/library
ollama list                # confirm it is present
curl $OLLAMA_BASE_URL/v1/models
```

Model tags in the Ollama library change over time and were not pinned for this document. Record
the exact tag and the digest shown by `ollama list` in the registry `version` field. Ollama models
are usually 4-bit quantised, so they are not identical to the 16-bit or FP8 weights served by vLLM;
do not mix the two for the same entry within one experiment.

### Determinism with local servers

The gateway sends temperature 0 and the configured seed. vLLM accepts a per-request `seed`; Ollama
accepts `seed` in its options. Batching on GPU servers can still change outputs slightly between
runs, which is one reason all results are cached and reused rather than regenerated.

If a model has a reasoning ("thinking") mode, decide whether it is on or off, set it the same way
for every Run, and record the setting in the registry entry.

## 4. Substituting an unavailable model

Model identifiers in the original plan were forward-dated. All of them were available when this
document was written, but models get retired, gated or priced out. Substitution is a configuration
change; no code changes.

1. **Add a new registry entry** in `configs/models.yaml` with a new key. Fill in every field:
   provider, exact model ID, version or snapshot date, family, context length, prices.
   Do not edit an existing entry in place, because past Run_Records refer to it.
2. **Point the role at the new key** in the experiment config (or the reference configuration).
3. **Keep the judge rule.** The judge family must still differ from every generator family.
4. **Re-index if the embedder changed.** Indexes are tied to the embedder model ID. Rebuilding the
   graph is needed only if the extraction model changed.
5. **Re-tune on dev if needed.** A new reranker or generator can shift the V4 settings and the R4
   escalation threshold. Re-tune on the dev split only; tuning on test is blocked.
6. **Run `uv run elevate verify-setup`**, then the relevant exit check.
7. **Record the substitution** under "Deviations" in the current phase report
   (`docs/phase-reports/phase-N.md`). Run_Records already store the exact model IDs and call dates.

Suggested fallbacks:

| If unavailable | Use | Notes |
|---|---|---|
| Qwen3.6-35B-A3B | `Qwen/Qwen3-30B-A3B-Instruct-2507` (Apache-2.0, verified on the Hub), or `Qwen/Qwen3.6-27B` | Same family; similar size class |
| Gemma 4 12B | `google/gemma-3-12b-it` | Gated: accept the Gemma licence on the Hub and set `HF_TOKEN` |
| Claude Sonnet 5.5 | the current Sonnet model on <https://platform.claude.com/docs/en/about-claude/models/overview> | Keep the family `anthropic` |
| No GPU for local generators | Any OpenAI-compatible hosted endpoint serving the same open weights: set `OPENAI_BASE_URL` or `VLLM_BASE_URL` to it | Record the host and its prices in the registry entry |
| `BAAI/bge-m3` too slow on CPU | `sentence-transformers/all-MiniLM-L6-v2` | Lower quality; report it as a deviation |
| `BAAI/bge-reranker-v2-m3` too slow on CPU | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lower quality; report it as a deviation |
| No OpenAI access for extraction or judge | Another API provider whose family differs from the generator, or a local model for extraction | The judge must still pass kappa ≥ 0.6 |

## 5. Provider-specific notes

- **Claude Sonnet 5.5 rejects sampling parameters.** Anthropic's model page states that setting
  `temperature`, `top_p` or `top_k` on this model returns a 400 error. The gateway must omit them
  for this entry, which departs from the temperature-0 rule (Requirement 3.3). Outputs are still
  reproducible through the cache. Record this in the Phase 0 report under "Deviations". The model
  also uses adaptive thinking by default; set the thinking/effort setting explicitly and freeze it.
- **Seeds.** OpenAI, vLLM and Ollama accept a seed; Anthropic does not. The gateway passes a seed
  only where the provider supports it (Requirement 3.9).
- **Prices change.** Re-check prices on the freeze date and do not change them afterwards for an
  experiment in progress; the Budget_Tracker and cost-per-1,000-queries figures depend on them.
- **Model IDs and call dates** are logged for every call and stored in every Run_Record, so a
  provider silently updating a model is at least visible in the records. Prefer dated snapshot IDs
  where the provider offers them.
