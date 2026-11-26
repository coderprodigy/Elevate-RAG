"""Vector_Store construction from configuration (Requirements 2.3, 2.5, 2.6)."""

from __future__ import annotations

from collections.abc import Callable

from elevate.config.schema import BackendSettings, VectorBackend
from elevate.core.errors import BackendError
from elevate.stores.vector.base import VectorStore
from elevate.stores.vector.faiss_store import FaissVectorStore, HnswParams, IndexType
from elevate.stores.vector.nano_store import NanoVectorStore

DEFAULT_VECTOR_BACKEND: VectorBackend = BackendSettings().vector
"""Backend used when nothing is configured: FAISS (Requirement 2.3)."""

VectorStoreFactory = Callable[[int, IndexType, HnswParams | None], VectorStore]

_FACTORIES: dict[str, VectorStoreFactory] = {
    "faiss": lambda dim, index_type, hnsw: FaissVectorStore(dim, index_type=index_type, hnsw=hnsw),
    "nanovectordb": lambda dim, _index_type, _hnsw: NanoVectorStore(dim),
}


def register_vector_backend(name: str, factory: VectorStoreFactory) -> None:
    """Make a backend available to :func:`create_vector_store` (used by the Phase 5 clients)."""
    _FACTORIES[name] = factory


def create_vector_store(
    settings: BackendSettings | None = None,
    *,
    dim: int,
    index_type: IndexType = "hnsw",
    hnsw: HnswParams | None = None,
    check_health: bool = True,
) -> VectorStore:
    """Build the configured Vector_Store, defaulting to FAISS when none is configured.

    When ``check_health`` is true the backend's health check runs before the store is
    returned, so a broken backend stops the run before indexing (Requirements 2.5, 2.6).
    ``index_type`` and ``hnsw`` apply to backends with configurable indexes (FAISS).
    """
    backend = (settings or BackendSettings()).vector
    factory = _FACTORIES.get(backend)
    if factory is None:
        raise BackendError(
            backend=backend,
            reason="no client for this vector backend is available in this build yet",
        )
    store = factory(dim, index_type, hnsw)
    if check_health:
        store.health_check()
    return store


__all__ = [
    "DEFAULT_VECTOR_BACKEND",
    "VectorStoreFactory",
    "create_vector_store",
    "register_vector_backend",
]
