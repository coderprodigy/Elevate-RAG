# 04 — Compute for open-weight models

Your Mac (16 GB unified memory) can run the **small** models: bge-small, Qwen3-Embedding-0.6B and
bge-reranker-v2-m3 on Apple MPS. That covers the reference embedder and reranker.

It **cannot** comfortably run Qwen3.6-35B-A3B (about 20 GB or more even at 4-bit), DeepSeek V4-Flash or
Qwen3-Embedding-8B. Gemma 4 12B fits at 4-bit, but answering thousands of questions would take many hours.

## Options (pick one per model, then record it in the model-freeze table)

| Option | Good for | Pros | Cons |
|---|---|---|---|
| **A. Hosted OpenAI-compatible endpoint** (OpenRouter, Together, Fireworks) — *recommended default* | Mid and small open generators, the open extractor | No setup; fast; pay per token; our gateway supports it | Costs money; **must pin the provider and quantisation** (see below) |
| B. University GPU (Waterloo ECE/CS servers, or a Digital Research Alliance of Canada allocation through your supervisor or instructor) | Everything open, including the 8B embedder | Free; full control | Queue times; you run vLLM yourself; account approval can take days |
| C. Local Ollama | Gemma 4 12B smoke tests only | Free, offline | Slow; quantisation differs from hosted versions |

## If you choose A (hosted)

The same open model can produce different outputs depending on which host and quantisation serves it.
To keep results comparable, the gateway sends OpenRouter provider-routing preferences: a fixed provider
order, `allow_fallbacks: false` and a required quantisation. Record the following in `configs/models.yaml`
for each open model:

```yaml
generator_mid:
  provider: openrouter
  model: <exact model slug>
  route: {order: [<provider>], allow_fallbacks: false, quantizations: [<fp8|bf16>]}
  price: {input_per_mtok: <...>, output_per_mtok: <...>}
```

Pick the provider on freeze day and don't change it afterwards.

## If you choose B (university GPU)

Tell me the GPU type and memory, and whether you can expose a port or only run batch jobs. I'll add a
`vllm serve` launch script and a SLURM batch template in `scripts/cluster/`. The gateway then talks to it
as an OpenAI-compatible endpoint (`OPENAI_COMPAT_BASE_URL`). Start the account request early, because
approval can take a while.
