# Docker database stacks

One Compose file per database (Requirement 2.4). None of them is needed for the no-setup
backends (FAISS/NanoVectorDB, NetworkX); start them only for Phase 5 and the DB benchmark.

| File | Service | Host port(s) | Client env var |
|------|---------|--------------|----------------|
| `qdrant.yml` | `qdrant/qdrant:v1.13.4` | 6333 (HTTP), 6334 (gRPC) | `QDRANT_URL`, `QDRANT_API_KEY` |
| `milvus.yml` | `milvusdb/milvus:v2.5.4` (+ etcd, MinIO) | 19530, 9091 (health) | `MILVUS_URI`, `MILVUS_TOKEN` |
| `pgvector.yml` | `pgvector/pgvector:0.8.0-pg16` | 5432 | `PGVECTOR_DSN` |
| `neo4j.yml` | `neo4j:5.26.2-community` | 7687 (Bolt), 7474 (HTTP) | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| `memgraph.yml` | `memgraph/memgraph:2.21.0` | 7688 (Bolt) | `MEMGRAPH_URI`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD` |
| `postgres-graph.yml` | `postgres:16.6` | 5433 | `POSTGRES_GRAPH_DSN` |

```bash
docker compose -f docker/qdrant.yml --env-file .env up -d     # start
docker compose -f docker/qdrant.yml ps                          # health status
docker compose -f docker/qdrant.yml down                        # stop, keep data
docker compose -f docker/qdrant.yml down -v                     # stop and delete volumes
```

## Resource limits (fairness rule, Requirement 26.3)

Every benchmarked database container gets the same limits:

- `ELEVATE_DB_CPUS` (default `2`)
- `ELEVATE_DB_MEMORY` (default `4g`)

Milvus's etcd and MinIO side services get `ELEVATE_DB_AUX_CPUS` (default `0.5`) and
`ELEVATE_DB_AUX_MEMORY` (default `512m`) each. Report them as Milvus overhead in the benchmark.

## Credentials

Defaults match `.env.example`. The Postgres stacks read separate parts that must agree with the
DSN the client uses:

- pgvector: `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_DB`, `PGVECTOR_PORT`
- Postgres graph: `POSTGRES_GRAPH_USER`, `POSTGRES_GRAPH_PASSWORD`, `POSTGRES_GRAPH_DB`, `POSTGRES_GRAPH_PORT`

Neo4j needs a password of at least 8 characters. Change every `change-me` default before
exposing any port beyond localhost.
