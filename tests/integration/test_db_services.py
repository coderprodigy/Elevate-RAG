"""Conformance tests for server-backed stores (FR-DB1, FR-DB2).

They run only against live services (see infrastructure/docker/docker-compose.yml) — set the
ELEVATE_* URL variables from .env.example. Without them every test is *skipped* (reported by
pytest -ra), never silently passed.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import uuid
from urllib.parse import urlparse

import numpy as np
import pytest

from elevate_rag.config import GraphStoreConfig, VectorStoreConfig
from elevate_rag.storage.graph import make_graph_store
from elevate_rag.storage.vector import make_vector_store, wait_until_searchable

pytestmark = pytest.mark.integration

VECTOR = {
    "qdrant": ("qdrant_client", "ELEVATE_QDRANT_URL"),
    "milvus": ("pymilvus", "ELEVATE_MILVUS_URI"),
    "pgvector": ("psycopg", "ELEVATE_PG_DSN"),
}
GRAPH = {
    "neo4j": ("neo4j", "ELEVATE_NEO4J_URI"),
    "memgraph": ("neo4j", "ELEVATE_MEMGRAPH_URI"),
    "postgres": ("psycopg", "ELEVATE_PG_DSN"),
}


def _require(module: str, env: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"package {module} not installed (uv sync --extra databases)")
    if not os.getenv(env):
        pytest.skip(f"{env} not set — start infrastructure/docker services to run")
    url = urlparse(os.environ[env])
    try:
        socket.create_connection((url.hostname or "localhost", url.port or 0), timeout=2).close()
    except OSError:
        pytest.skip(f"{env}={url.hostname}:{url.port} not reachable — service not started")


@pytest.mark.parametrize("backend", sorted(VECTOR))
def test_vector_store_conformance(backend: str) -> None:
    _require(*VECTOR[backend])
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((200, 32)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"c{i:03d}" for i in range(200)]
    store = make_vector_store(
        VectorStoreConfig(backend=backend, collection_prefix="elevtest"),
        f"t{uuid.uuid4().hex[:8]}",
        32,
    )
    try:
        store.add(ids, vecs, [{"source": "a" if i % 2 else "b"} for i in range(200)])
        assert store.count() == 200
        assert store.search(vecs[17], 1)[0][0] == "c017"
        assert all(int(h[0][1:]) % 2 for h in store.search(vecs[17], 5, {"source": "a"}))
        assert wait_until_searchable(store, "probe", vecs[3]) < 30
    finally:
        getattr(store, "drop", lambda: None)()
        store.close()


@pytest.mark.parametrize("backend", sorted(GRAPH))
def test_graph_store_conformance(backend: str) -> None:
    _require(*GRAPH[backend])
    store = make_graph_store(GraphStoreConfig(backend=backend))  # type: ignore[arg-type]
    try:
        store.clear()
        store.bulk_load(
            [{"id": x, "type": "t", "description": x} for x in "ABCD"],
            [
                {"src": "A", "tgt": "B", "description": "", "weight": 1.0},
                {"src": "B", "tgt": "C", "description": "", "weight": 1.0},
                {"src": "C", "tgt": "D", "description": "", "weight": 1.0},
            ],
        )
        assert store.counts() == (4, 3)
        assert store.get_entity("A") is not None and store.get_entity("Z") is None
        assert store.neighbors("A", 1) == {"B"} and store.neighbors("A", 2) == {"B", "C"}
    finally:
        store.clear()
        store.close()
