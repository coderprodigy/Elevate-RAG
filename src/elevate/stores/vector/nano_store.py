"""NanoVectorDB Vector_Store: the alternative no-setup vector backend (Requirements 2.1, 2.3).

NanoVectorDB does exact cosine search over an in-memory numpy matrix, the same store LightRAG
uses by default. The store is in-memory only: it points NanoVectorDB at a fresh, never-written
storage path so no file from an earlier run is ever loaded.
"""

from __future__ import annotations

import importlib
import tempfile
import uuid
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

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

_SOURCE = "source"


def load_nano_vectordb() -> ModuleType:
    """Import NanoVectorDB, turning an import failure into a ``BackendError`` (Req 2.6)."""
    try:
        return importlib.import_module("nano_vectordb")
    except ImportError as exc:  # pragma: no cover - nano-vectordb is a core dependency
        raise BackendError(
            backend="nanovectordb",
            reason=f"cannot import nano_vectordb ({exc})",
            setup_section=EMBEDDED_SETUP_SECTION,
        ) from exc


def _nano_version() -> str:
    try:
        return metadata.version("nano-vectordb")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed with the core deps
        return "unknown"


def _unused_storage_path() -> str:
    return str(Path(tempfile.gettempdir()) / f"elevate-nanovdb-{uuid.uuid4().hex}.json")


class NanoVectorStore(BaseVectorStore):
    """In-process NanoVectorDB backend (exact cosine search)."""

    name = "nanovectordb"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._nano = load_nano_vectordb()
        self._db: Any = self._nano.NanoVectorDB(dim, storage_file=_unused_storage_path())

    def _backend_upsert(self, items: list[VectorItem], matrix: FloatMatrix) -> None:
        self._db.upsert(
            [
                {"__id__": item.id, "__vector__": row, _SOURCE: item.source}
                for item, row in zip(items, matrix, strict=True)
            ]
        )

    def _backend_delete(self, ids: list[str]) -> None:
        self._db.delete(ids)

    def _backend_search(
        self, query: FloatMatrix, k: int, source: str | None
    ) -> list[tuple[str, float]]:
        def keep(data: dict[str, Any]) -> bool:
            return bool(data[_SOURCE] == source)

        results = self._db.query(query, top_k=k, filter_lambda=None if source is None else keep)
        return [(str(r["__id__"]), finite_score(r["__metrics__"])) for r in results]

    def _backend_params(self) -> dict[str, Any]:
        return {"index_type": "flat", "backend_version": _nano_version()}

    def health_check(self) -> HealthStatus:
        """Insert and query a probe vector in a throwaway NanoVectorDB instance."""
        try:
            probe = self._nano.NanoVectorDB(2, storage_file=_unused_storage_path())
            probe.upsert(
                [{"__id__": "probe", "__vector__": np.array([1.0, 0.0], dtype=np.float32)}]
            )
            found = probe.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
            if not found or found[0]["__id__"] != "probe":
                raise RuntimeError("probe search returned the wrong neighbour")
        except Exception as exc:
            raise BackendError(
                backend=self.name, reason=str(exc), setup_section=EMBEDDED_SETUP_SECTION
            ) from exc
        return HealthStatus(backend=self.name, version=_nano_version())


__all__ = ["NanoVectorStore", "load_nano_vectordb"]
