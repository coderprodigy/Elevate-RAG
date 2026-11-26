"""Select the Graph_Store backend from configuration (Req 2.3, 2.5, 2.6)."""

from __future__ import annotations

from pathlib import Path

from elevate.config.schema import BackendSettings, GraphBackend
from elevate.core.errors import BackendError
from elevate.stores.graph.base import GraphStore
from elevate.stores.graph.networkx_store import NetworkxGraphStore

_DATABASE_BACKENDS: dict[str, str] = {
    "neo4j": "Neo4j",
    "memgraph": "Memgraph",
    "postgres": "Postgres graph tables",
}


def create_graph_store(
    settings: BackendSettings | GraphBackend | None = None,
    *,
    path: str | Path | None = None,
    check_health: bool = True,
) -> GraphStore:
    """Build the configured graph backend; NetworkX when nothing else is configured.

    For NetworkX, an existing ``path`` is loaded (see :meth:`NetworkxGraphStore.load`);
    otherwise an empty store is returned. With ``check_health`` the backend's health check
    runs before the store is returned, so a failing backend stops the run before indexing.
    """
    backend: str
    if settings is None:
        backend = "networkx"
    elif isinstance(settings, BackendSettings):
        backend = settings.graph
    else:
        backend = settings

    store: GraphStore
    if backend == "networkx":
        if path is not None and Path(path).exists():
            store = NetworkxGraphStore.load(path)
        else:
            store = NetworkxGraphStore()
    elif backend in _DATABASE_BACKENDS:
        raise BackendError(
            backend=_DATABASE_BACKENDS[backend],
            reason="this graph backend is not implemented yet (planned for Phase 5); "
            "use backends.graph: networkx",
            setup_section=backend,
        )
    else:
        raise ValueError(f"Unknown graph backend '{backend}'")

    if check_health:
        store.health_check()
    return store


__all__ = ["create_graph_store"]
