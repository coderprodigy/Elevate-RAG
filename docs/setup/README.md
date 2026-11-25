# Setup checklist

These are the things only you (or your team) can do. Work through them in order. Each item says which
implementation phase it blocks.

| # | Task | Blocks | Guide | Done |
|---|---|---|---|---|
| 1 | Install `uv`, check git, optionally create a GitHub repo | Phase 0 | [01_local_environment.md](01_local_environment.md) | ☑ uv + git done |
| 2 | *Optional:* a Docker runtime on the Mac (OrbStack). The main plan runs the databases on the Waterloo server | Day 4+ (Qdrant/Neo4j) | [01_local_environment.md](01_local_environment.md) | ☐ |
| 3 | Create API accounts and keys, put them in `.env` | Phase 0 | [02_accounts_api_keys_and_models.md](02_accounts_api_keys_and_models.md) | ☐ |
| 4 | Agree on the frozen model list and a spending cap as a team | Phase 0 → 2 | [02_accounts_api_keys_and_models.md](02_accounts_api_keys_and_models.md) | ☐ |
| 5 | Datasets: now fully automatic (`hgrag data download`) | Phase 1 | [03_datasets.md](03_datasets.md) | ☑ done |
| 6 | Get Waterloo GPU server access and run the probe script (hosted endpoint is the fallback) | Day 4: Phase 3 generator, Phase 4 | [05_waterloo_servers.md](05_waterloo_servers.md) · [04](04_compute_for_open_models.md) | ☐ |
| 7 | Line up two people for human labelling (100 answers, 50 triples, about 3 hours each) | Phases 3–4 | [IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md) | ☐ |

Phase 0 is built and tested **without** API keys (it uses a fake provider). Keys (item 3) are needed for the first real `hgrag llm ping`, and the Waterloo server (item 6) is needed on day 4.
