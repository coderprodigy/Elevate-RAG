"""Graph store adapters for the E4 Level A graph benchmark (NetworkX, Neo4j, Memgraph, Postgres).

All stores load the same entity graph (undirected neighbourhoods) and answer the same three
operations: lookup by name, 1-hop and 2-hop neighbourhood. Neo4j and Memgraph share the Bolt
client and the same Cypher; Postgres uses plain tables to test whether a graph DB is needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import networkx as nx

Edge = tuple[str, str]


class GraphStore(Protocol):
    name: str

    def load(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None: ...

    def lookup(self, name: str) -> bool: ...

    def neighbours(self, name: str, hops: int) -> set[str]: ...

    def clear(self) -> None: ...


class NetworkXStore:
    name = "networkx"

    def __init__(self) -> None:
        self._graph: Any = nx.Graph()

    def load(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None:
        self._graph = nx.Graph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_edges_from(edges)

    def lookup(self, name: str) -> bool:
        return bool(self._graph.has_node(name))

    def neighbours(self, name: str, hops: int) -> set[str]:
        if not self._graph.has_node(name):
            return set()
        reach = nx.single_source_shortest_path_length(self._graph, name, cutoff=hops)
        return {n for n, d in reach.items() if 0 < d <= hops}

    def clear(self) -> None:
        self._graph = nx.Graph()


class BoltStore:
    """Neo4j or Memgraph through the Bolt protocol (same Cypher for both)."""

    def __init__(self, name: str, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self.name = name
        self._driver = GraphDatabase.driver(uri, auth=(user, password) if user else None)

    @classmethod
    def neo4j(cls) -> BoltStore:
        env = os.environ.get
        return cls(
            "neo4j",
            env("NEO4J_URI", "bolt://localhost:7687"),
            env("NEO4J_USERNAME", "neo4j"),
            env("NEO4J_PASSWORD", "hgrag-dev-password"),
        )

    @classmethod
    def memgraph(cls) -> BoltStore:
        return cls("memgraph", os.environ.get("MEMGRAPH_URI", "bolt://localhost:7688"), "", "")

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self._driver.session() as session:
            return list(session.run(query, **params))

    def load(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None:
        self.clear()
        self._run(
            "CREATE INDEX ON :Entity(name)"
            if self.name == "memgraph"
            else "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)"
        )
        for start in range(0, len(nodes), 5000):
            self._run(
                "UNWIND $names AS n CREATE (:Entity {name: n})",
                names=list(nodes[start : start + 5000]),
            )
        for start in range(0, len(edges), 5000):
            self._run(
                "UNWIND $edges AS e MATCH (a:Entity {name: e[0]}), (b:Entity {name: e[1]}) "
                "CREATE (a)-[:REL]->(b)",
                edges=[list(e) for e in edges[start : start + 5000]],
            )

    def lookup(self, name: str) -> bool:
        return bool(self._run("MATCH (n:Entity {name: $n}) RETURN n LIMIT 1", n=name))

    def neighbours(self, name: str, hops: int) -> set[str]:
        rows = self._run(
            f"MATCH (n:Entity {{name: $n}})-[*1..{int(hops)}]-(m:Entity) "
            "WHERE m.name <> $n RETURN DISTINCT m.name AS name",
            n=name,
        )
        return {row["name"] for row in rows}

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")


class PostgresGraphStore:
    """Nodes + edges tables; neighbourhoods by joins (tests "is a graph DB needed at all?")."""

    name = "postgres"

    def __init__(self, dsn: str | None = None) -> None:
        import psycopg

        from hgrag.bench.vector_stores import _postgres_dsn

        self._conn = psycopg.connect(dsn or _postgres_dsn(), autocommit=True)

    def load(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS g_edges; DROP TABLE IF EXISTS g_nodes")
            cur.execute("CREATE TABLE g_nodes (name text PRIMARY KEY)")
            cur.execute("CREATE TABLE g_edges (a text, b text)")
            with cur.copy("COPY g_nodes (name) FROM STDIN") as copy:
                for node in nodes:
                    copy.write_row((node,))
            with cur.copy("COPY g_edges (a, b) FROM STDIN") as copy:
                for a, b in edges:
                    copy.write_row((a, b))
                    copy.write_row((b, a))  # store both directions: undirected neighbourhoods
            cur.execute("CREATE INDEX ON g_edges (a)")

    def lookup(self, name: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM g_nodes WHERE name = %s", (name,))
            return cur.fetchone() is not None

    def neighbours(self, name: str, hops: int) -> set[str]:
        sql = "SELECT DISTINCT b FROM g_edges WHERE a = %s"
        params: tuple[str, ...] = (name,)
        if hops >= 2:
            sql = (
                "SELECT DISTINCT e2.b FROM g_edges e1 JOIN g_edges e2 ON e2.a = e1.b "
                "WHERE e1.a = %s UNION SELECT b FROM g_edges WHERE a = %s"
            )
            params = (name, name)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return {row[0] for row in cur.fetchall()} - {name}

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS g_edges; DROP TABLE IF EXISTS g_nodes")


GRAPH_STORES = ("networkx", "neo4j", "memgraph", "postgres")


def make_graph_store(name: str) -> GraphStore:
    if name == "networkx":
        return NetworkXStore()
    if name == "neo4j":
        return BoltStore.neo4j()
    if name == "memgraph":
        return BoltStore.memgraph()
    if name == "postgres":
        return PostgresGraphStore()
    raise KeyError(f"unknown graph store '{name}'; choose from {GRAPH_STORES}")
