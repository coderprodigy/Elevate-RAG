"""FAISS Vector_Store: the default no-setup vector backend (Requirements 2.1, 2.3).

Two index types are supported:

- ``hnsw``: ``IndexHNSWFlat`` with inner-product metric and the configured ``M``,
  ``ef_construction`` and ``ef_search`` (recorded by :meth:`FaissVectorStore.index_params`).
- ``flat``: exact ``IndexFlatIP`` search, the recall reference for HNSW tuning.

FAISS HNSW indexes cannot remove vectors, so the store keeps the normalised vectors itself.
New IDs are added to the live index incrementally; replacing or deleting an ID marks the index
stale and it is rebuilt from the kept vectors, in insertion order, before the next search. The
build is deterministic: FAISS HNSW uses a fixed level-assignment seed, and vectors are added
single-threaded. Multithreaded HNSW insertion depends on thread scheduling, and on macOS it can
deadlock when another OpenMP runtime is loaded in the same process.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np

from elevate.core.errors import BackendError
from elevate.stores.vector.base import (
    EMBEDDED_SETUP_SECTION,
    BaseVectorStore,
    FloatMatrix,
    HealthStatus,
    VectorItem,
    finite_score,
)

IndexType = Literal["hnsw", "flat"]


@dataclass(frozen=True)
class HnswParams:
    """HNSW build and query parameters (Requirement 11.3)."""

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64

    def __post_init__(self) -> None:
        for label, value in (
            ("M", self.m),
            ("ef_construction", self.ef_construction),
            ("ef_search", self.ef_search),
        ):
            if value < 1:
                raise ValueError(f"HNSW {label} must be at least 1, got {value}")


def load_faiss() -> ModuleType:
    """Import FAISS, turning an import failure into a ``BackendError`` (Requirement 2.6)."""
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:  # pragma: no cover - faiss-cpu is a core dependency
        raise BackendError(
            backend="faiss",
            reason=f"cannot import faiss ({exc})",
            setup_section=EMBEDDED_SETUP_SECTION,
        ) from exc


class FaissVectorStore(BaseVectorStore):
    """In-process FAISS backend with cosine similarity."""

    name = "faiss"

    def __init__(
        self,
        dim: int,
        *,
        index_type: IndexType = "hnsw",
        hnsw: HnswParams | None = None,
    ) -> None:
        super().__init__(dim)
        if index_type not in ("hnsw", "flat"):
            raise ValueError(f"index_type must be 'hnsw' or 'flat', got {index_type!r}")
        self._faiss = load_faiss()
        self.index_type: IndexType = index_type
        self.hnsw = hnsw or HnswParams()
        self._vectors: dict[str, FloatMatrix] = {}
        self._positions: list[str] = []  # FAISS internal id -> item id
        self._index: Any = None
        self._stale = True

    # ------------------------------------------------------------------ index management
    def _new_index(self) -> Any:
        faiss = self._faiss
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)
        index = faiss.IndexHNSWFlat(self.dim, self.hnsw.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw.ef_construction
        index.hnsw.efSearch = self.hnsw.ef_search
        return index

    def _ensure_index(self) -> Any:
        if self._stale or self._index is None:
            index = self._new_index()
            self._positions = list(self._vectors)
            if self._positions:
                self._add(index, np.stack([self._vectors[i] for i in self._positions]))
            self._index = index
            self._stale = False
        return self._index

    def _add(self, index: Any, matrix: FloatMatrix) -> None:
        faiss = self._faiss
        threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            index.add(matrix)
        finally:
            faiss.omp_set_num_threads(threads)

    # ------------------------------------------------------------------ backend hooks
    def _backend_upsert(self, items: list[VectorItem], matrix: FloatMatrix) -> None:
        replaces = any(item.id in self._vectors for item in items)
        for item, row in zip(items, matrix, strict=True):
            self._vectors[item.id] = row
        if replaces or self._stale or self._index is None:
            self._stale = True
            return
        self._add(self._index, matrix)
        self._positions.extend(item.id for item in items)

    def _backend_delete(self, ids: list[str]) -> None:
        for item_id in ids:
            del self._vectors[item_id]
        self._stale = True

    def _backend_search(
        self, query: FloatMatrix, k: int, source: str | None
    ) -> list[tuple[str, float]]:
        faiss = self._faiss
        index = self._ensure_index()
        selector = None
        if source is not None:
            wanted = set(self._ids_with_source(source))
            positions = [p for p, item_id in enumerate(self._positions) if item_id in wanted]
            selector = faiss.IDSelectorBatch(np.asarray(positions, dtype=np.int64))
        if self.index_type == "hnsw":
            # efSearch below k would cap the result count; widen it for this query only.
            params = faiss.SearchParametersHNSW(efSearch=max(self.hnsw.ef_search, k))
        else:
            params = faiss.SearchParameters()
        if selector is not None:
            params.sel = selector
        k_eff = min(k, len(self._positions))
        scores, idx = index.search(query.reshape(1, -1), k_eff, params=params)
        return [
            (self._positions[int(p)], finite_score(s))
            for s, p in zip(scores[0], idx[0], strict=True)
            if p >= 0
        ]

    def _backend_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "index_type": self.index_type,
            "backend_version": self._faiss.__version__,
        }
        if self.index_type == "hnsw":
            params.update(
                M=self.hnsw.m,
                ef_construction=self.hnsw.ef_construction,
                ef_search=self.hnsw.ef_search,
            )
        return params

    def health_check(self) -> HealthStatus:
        """Build and query a two-vector index to prove FAISS works in this environment."""
        try:
            faiss = self._faiss
            probe = faiss.IndexFlatIP(2)
            probe.add(np.eye(2, dtype=np.float32))
            _, idx = probe.search(np.array([[1.0, 0.0]], dtype=np.float32), 1)
            if int(idx[0][0]) != 0:
                raise RuntimeError("probe search returned the wrong neighbour")
            version = str(faiss.__version__)
        except Exception as exc:
            raise BackendError(
                backend=self.name, reason=str(exc), setup_section=EMBEDDED_SETUP_SECTION
            ) from exc
        return HealthStatus(
            backend=self.name, version=version, detail=f"index_type={self.index_type}"
        )


__all__ = ["FaissVectorStore", "HnswParams", "IndexType", "load_faiss"]
