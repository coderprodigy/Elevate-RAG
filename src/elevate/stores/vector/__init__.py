"""Vector_Store protocol, no-setup backends (FAISS, NanoVectorDB) and the backend factory."""

from elevate.stores.vector.base import HealthStatus, Hit, VectorItem, VectorStore
from elevate.stores.vector.factory import (
    DEFAULT_VECTOR_BACKEND,
    create_vector_store,
    register_vector_backend,
)
from elevate.stores.vector.faiss_store import FaissVectorStore, HnswParams
from elevate.stores.vector.nano_store import NanoVectorStore

__all__ = [
    "DEFAULT_VECTOR_BACKEND",
    "FaissVectorStore",
    "HealthStatus",
    "Hit",
    "HnswParams",
    "NanoVectorStore",
    "VectorItem",
    "VectorStore",
    "create_vector_store",
    "register_vector_backend",
]
