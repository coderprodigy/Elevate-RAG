# Database stacks (Phase 6 / E4)

One stack at a time on a 16 GB machine; on the Waterloo server they can run together.
Every engine gets **the same limits** (`HGRAG_DB_CPUS`, `HGRAG_DB_MEM`, default 4 CPUs / 4 GB) and
binds to `127.0.0.1` only (reach it through the SSH tunnel, docs/setup/05_waterloo_servers.md).

```bash
docker compose -f docker/compose.core.yaml up -d                     # Qdrant + Neo4j (reference)
docker compose -f docker/compose.bench-vector.yaml --profile milvus up -d
docker compose -f docker/compose.bench-vector.yaml --profile pgvector up -d
docker compose -f docker/compose.bench-graph.yaml --profile memgraph up -d
docker compose -f docker/compose.bench-graph.yaml --profile postgres up -d
uv run hgrag bench health                                             # which stores answer
```

Image tags are pinned. Before the first run on the server, check they are still available and
record the final versions in the report appendix (`docker compose ... images`).
Credentials are development defaults for a localhost-only service; change them in `.env` if the
server is shared (`NEO4J_PASSWORD`, `POSTGRES_PASSWORD`).
