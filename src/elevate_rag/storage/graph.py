"""Graph store adapters for the Level-A graph micro-benchmark (FR-DB2, FR-DB5, D-05).

The *experiments* use LightRAG's own graph storages (selected by `lightrag.graph_storage`); these
adapters exist so each database can be timed in isolation on the same exported graph:
bulk load, entity lookup by name, and 1-/2-hop neighbourhood fetches.

Neo4j and Memgraph share one Bolt/Cypher implementation.
"""

from __future__ import annotations

import os
from typing import Any

import networkx as nx

from elevate_rag.config import GraphStoreConfig
from elevate_rag.utils import ConfigError, StorageError

Node = dict[str, Any]  # {"id": str, "type": str, "description": str}
Edge = dict[str, Any]  # {"src": str, "tgt": str, "description": str, "weight": float}


class GraphStore:
    backend = "abstract"

    def bulk_load(self, nodes: list[Node], edges: list[Edge]) -> None:
        raise NotImplementedError

    def get_entity(self, name: str) -> Node | None:
        raise NotImplementedError

    def neighbors(self, name: str, hops: int = 1) -> set[str]:
        raise NotImplementedError

    def counts(self) -> tuple[int, int]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NetworkXStore(GraphStore):
    backend = "networkx"

    def __init__(self, cfg: GraphStoreConfig | None = None) -> None:
        self.g = nx.Graph()

    def bulk_load(self, nodes: list[Node], edges: list[Edge]) -> None:
        self.g.add_nodes_from((n["id"], {k: v for k, v in n.items() if k != "id"}) for n in nodes)
        self.g.add_edges_from(
            (e["src"], e["tgt"], {k: v for k, v in e.items() if k not in ("src", "tgt")})
            for e in edges
        )

    def get_entity(self, name: str) -> Node | None:
        return {"id": name, **self.g.nodes[name]} if name in self.g else None

    def neighbors(self, name: str, hops: int = 1) -> set[str]:
        if name not in self.g:
            return set()
        lengths = nx.single_source_shortest_path_length(self.g, name, cutoff=hops)
        return {n for n, d in lengths.items() if 0 < d <= hops}

    def counts(self) -> tuple[int, int]:
        return self.g.number_of_nodes(), self.g.number_of_edges()

    def clear(self) -> None:
        self.g.clear()


class BoltStore(GraphStore):
    """Neo4j / Memgraph via the official `neo4j` driver (Cypher)."""

    def __init__(self, cfg: GraphStoreConfig) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ConfigError("install the `neo4j` extra") from exc
        self.backend = cfg.backend
        default = "ELEVATE_NEO4J_URI" if cfg.backend == "neo4j" else "ELEVATE_MEMGRAPH_URI"
        uri = os.getenv(cfg.url_env or default)
        if not uri:
            raise ConfigError(f"{cfg.backend}: set {cfg.url_env or default}")
        user = os.getenv(cfg.user_env or "ELEVATE_NEO4J_USER", "")
        pwd = os.getenv(cfg.password_env or "ELEVATE_NEO4J_PASSWORD", "")
        auth = (user, pwd) if cfg.backend == "neo4j" else None
        self._driver = GraphDatabase.driver(uri, auth=auth)
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            raise StorageError(f"{cfg.backend} unreachable at {uri}: {exc}") from exc
        index = (
            "CREATE INDEX elevate_entity_id IF NOT EXISTS FOR (n:ElevateEntity) ON (n.id)"
            if cfg.backend == "neo4j"
            else "CREATE INDEX ON :ElevateEntity(id)"
        )
        try:
            self._run(index)
        except Exception as exc:  # Memgraph raises if the index exists
            if "already exists" not in str(exc).lower():
                raise

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self._driver.session() as s:
            return list(s.run(query, **params))

    def bulk_load(self, nodes: list[Node], edges: list[Edge], batch: int = 1000) -> None:
        for i in range(0, len(nodes), batch):
            self._run(
                "UNWIND $rows AS r MERGE (n:ElevateEntity {id: r.id}) "
                "SET n.type = r.type, n.description = r.description",
                rows=nodes[i : i + batch],
            )
        for i in range(0, len(edges), batch):
            self._run(
                "UNWIND $rows AS r MATCH (a:ElevateEntity {id: r.src}) "
                "MATCH (b:ElevateEntity {id: r.tgt}) MERGE (a)-[e:REL]->(b) "
                "SET e.description = r.description, e.weight = r.weight",
                rows=edges[i : i + batch],
            )

    def get_entity(self, name: str) -> Node | None:
        rows = self._run("MATCH (n:ElevateEntity {id: $id}) RETURN n", id=name)
        return dict(rows[0]["n"]) if rows else None

    def neighbors(self, name: str, hops: int = 1) -> set[str]:
        q = (
            f"MATCH (n:ElevateEntity {{id: $id}})-[*1..{int(hops)}]-(m:ElevateEntity) "
            "WHERE m.id <> $id RETURN DISTINCT m.id AS id"
        )
        return {r["id"] for r in self._run(q, id=name)}

    def counts(self) -> tuple[int, int]:
        n = self._run("MATCH (n:ElevateEntity) RETURN count(n) AS c")[0]["c"]
        e = self._run("MATCH (:ElevateEntity)-[r:REL]->() RETURN count(r) AS c")[0]["c"]
        return int(n), int(e)

    def clear(self) -> None:
        self._run("MATCH (n:ElevateEntity) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()


class PostgresGraphStore(GraphStore):
    """Plain relational tables + recursive CTE — tests whether a graph DB is needed at all."""

    backend = "postgres"

    def __init__(self, cfg: GraphStoreConfig) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ConfigError("install the `postgres` extra") from exc
        dsn = os.getenv(cfg.url_env or "ELEVATE_PG_DSN")
        if not dsn:
            raise ConfigError("postgres graph store: set ELEVATE_PG_DSN")
        try:
            self._conn = psycopg.connect(dsn, autocommit=True)
        except psycopg.OperationalError as exc:
            raise StorageError(f"PostgreSQL unreachable: {exc}") from exc
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS elevate_nodes (id text PRIMARY KEY, "
            "type text, description text)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS elevate_edges (src text, tgt text, "
            "description text, weight double precision, PRIMARY KEY (src, tgt))"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS elevate_edges_tgt ON elevate_edges (tgt)")

    def bulk_load(self, nodes: list[Node], edges: list[Edge]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO elevate_nodes VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                [(n["id"], n.get("type"), n.get("description")) for n in nodes],
            )
            cur.executemany(
                "INSERT INTO elevate_edges VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(e["src"], e["tgt"], e.get("description"), e.get("weight", 1.0)) for e in edges],
            )

    def get_entity(self, name: str) -> Node | None:
        r = self._conn.execute(
            "SELECT id, type, description FROM elevate_nodes WHERE id = %s", (name,)
        ).fetchone()
        return {"id": r[0], "type": r[1], "description": r[2]} if r else None

    def neighbors(self, name: str, hops: int = 1) -> set[str]:
        rows = self._conn.execute(
            """WITH RECURSIVE und(a, b) AS (SELECT src, tgt FROM elevate_edges
                                            UNION ALL SELECT tgt, src FROM elevate_edges),
               walk(id, depth) AS (SELECT %s::text, 0
                                   UNION SELECT und.b, walk.depth + 1 FROM walk
                                   JOIN und ON und.a = walk.id WHERE walk.depth < %s)
               SELECT DISTINCT id FROM walk WHERE id <> %s""",
            (name, hops, name),
        ).fetchall()
        return {r[0] for r in rows}

    def counts(self) -> tuple[int, int]:
        return self._scalar("SELECT count(*) FROM elevate_nodes"), self._scalar(
            "SELECT count(*) FROM elevate_edges"
        )

    def _scalar(self, sql: str) -> int:
        row = self._conn.execute(sql).fetchone()
        if row is None:
            raise StorageError(f"no row returned for {sql!r}")
        return int(row[0])

    def clear(self) -> None:
        self._conn.execute("TRUNCATE elevate_nodes, elevate_edges")

    def close(self) -> None:
        self._conn.close()


def make_graph_store(cfg: GraphStoreConfig) -> GraphStore:
    if cfg.backend == "networkx":
        return NetworkXStore(cfg)
    if cfg.backend in ("neo4j", "memgraph"):
        return BoltStore(cfg)
    if cfg.backend == "postgres":
        return PostgresGraphStore(cfg)
    raise ConfigError(f"unknown graph backend {cfg.backend!r}")
