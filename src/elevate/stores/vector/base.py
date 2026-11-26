"""Vector_Store protocol and shared behaviour for every vector backend (Requirement 2.1).

Every backend exposes the same operations: insert/update (:meth:`VectorStore.upsert`),
:meth:`~VectorStore.delete`, top-k :meth:`~VectorStore.search` with an optional filter on the
chunk ``source``, :meth:`~VectorStore.index_params` for Run_Record reporting and
:meth:`~VectorStore.health_check` (Requirements 2.5, 2.6).

Similarity is cosine for every backend: vectors are L2-normalised on insert and on query, and
the score is the inner product of the normalised vectors (higher is better). Each stored vector
carries the embedding model name and version that produced it (Requirement 11.1); one index
holds vectors from one embedding model only.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from elevate.core.errors import EmbeddingMismatchError

Vector = Sequence[float]
"""A dense embedding as a sequence of floats (lists, tuples and 1-D numpy arrays all fit)."""

FloatMatrix = npt.NDArray[np.float32]

EMBEDDED_SETUP_SECTION = "3-installation"
"""Setup_Docs anchor for the embedded backends: they fail only when the install is broken."""

HealthState = Literal["ok"]


@dataclass(frozen=True)
class VectorItem:
    """One vector to store, tagged with its source and embedding model (Requirement 11.1)."""

    id: str
    vector: Vector
    source: str
    model_id: str
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """One search result. ``score`` is cosine similarity; results are sorted by it, descending."""

    id: str
    score: float
    source: str
    model_id: str
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthStatus:
    """Outcome of a successful health check (Requirement 2.5)."""

    backend: str
    version: str
    status: HealthState = "ok"
    detail: str = ""

    def summary(self) -> str:
        """One-line report: backend name, version and status."""
        line = f"{self.backend} {self.version}: {self.status}"
        return f"{line} ({self.detail})" if self.detail else line


@runtime_checkable
class VectorStore(Protocol):
    """The interface every vector backend implements (Requirement 2.1)."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def embedding_model(self) -> tuple[str, str | None] | None:
        """``(model_id, model_version)`` of the stored vectors, or ``None`` when empty."""
        ...

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """Insert new items and replace items whose ``id`` already exists."""
        ...

    def delete(self, ids: Iterable[str]) -> int:
        """Remove the given IDs (unknown IDs are ignored); return how many were removed."""
        ...

    def search(self, q: Vector, k: int, source: str | None = None) -> list[Hit]:
        """Return up to ``k`` nearest items, restricted to ``source`` when given."""
        ...

    def ids(self) -> set[str]:
        """All stored IDs (used by the shared-chunk invariant, Requirement 11.5)."""
        ...

    def __len__(self) -> int: ...

    def index_params(self) -> dict[str, Any]:
        """Index configuration for the Run_Record (Requirements 2.1, 11.3)."""
        ...

    def health_check(self) -> HealthStatus:
        """Report name, version and status, or raise ``BackendError`` (Requirements 2.5, 2.6)."""
        ...


@dataclass(frozen=True)
class _Record:
    """Per-item bookkeeping shared by the backends (everything except the vector)."""

    source: str
    model_id: str
    model_version: str | None
    metadata: Mapping[str, Any]

    def hit(self, item_id: str, score: float) -> Hit:
        return Hit(
            id=item_id,
            score=score,
            source=self.source,
            model_id=self.model_id,
            model_version=self.model_version,
            metadata=self.metadata,
        )


class BaseVectorStore(ABC):
    """Validation and metadata bookkeeping shared by concrete backends.

    Subclasses implement the ``_backend_*`` hooks; this class validates inputs, enforces
    one embedding model per index and keeps the ``id -> source/model/metadata`` table.
    """

    name: str = "base"

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        self._dim = dim
        self._records: dict[str, _Record] = {}
        self._model: tuple[str, str | None] | None = None

    # ------------------------------------------------------------------ public API
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embedding_model(self) -> tuple[str, str | None] | None:
        return self._model

    def upsert(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        batch: dict[str, VectorItem] = {}
        for item in items:
            if not item.id:
                raise ValueError("vector item id must be non-empty")
            batch[item.id] = item  # last occurrence of a duplicate id wins
        model = self._check_model(batch.values())
        matrix = self._normalise_rows([item.vector for item in batch.values()])
        self._backend_upsert(list(batch.values()), matrix)
        for item in batch.values():
            self._records[item.id] = _Record(
                source=item.source,
                model_id=item.model_id,
                model_version=item.model_version,
                metadata=dict(item.metadata),
            )
        self._model = model

    def delete(self, ids: Iterable[str]) -> int:
        present = [i for i in dict.fromkeys(ids) if i in self._records]
        if not present:
            return 0
        self._backend_delete(present)
        for item_id in present:
            del self._records[item_id]
        if not self._records:
            self._model = None
        return len(present)

    def search(self, q: Vector, k: int, source: str | None = None) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        query = self._normalise_rows([q])[0]
        if not self._records:
            return []
        if source is not None and not any(r.source == source for r in self._records.values()):
            return []
        scored = self._backend_search(query, k, source)
        return [self._records[item_id].hit(item_id, score) for item_id, score in scored]

    def ids(self) -> set[str]:
        return set(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def index_params(self) -> dict[str, Any]:
        model_id, model_version = self._model or (None, None)
        return {
            "backend": self.name,
            "dim": self._dim,
            "metric": "cosine",
            "count": len(self._records),
            "embedding_model_id": model_id,
            "embedding_model_version": model_version,
            **self._backend_params(),
        }

    @abstractmethod
    def health_check(self) -> HealthStatus: ...

    # ------------------------------------------------------------------ helpers
    def _ids_with_source(self, source: str) -> list[str]:
        return [i for i, r in self._records.items() if r.source == source]

    def _check_model(self, items: Iterable[VectorItem]) -> tuple[str, str | None]:
        model = self._model
        for item in items:
            if not item.model_id:
                raise ValueError(f"vector item {item.id!r} has an empty model_id")
            current = (item.model_id, item.model_version)
            if model is None:
                model = current
            elif current != model:
                raise EmbeddingMismatchError(
                    query_model=_model_label(current), index_model=_model_label(model)
                )
        assert model is not None
        return model

    def _normalise_rows(self, vectors: Sequence[Vector]) -> FloatMatrix:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dim:
            raise ValueError(f"expected vectors of dimension {self._dim}, got shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must contain only finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero vectors have no cosine similarity and cannot be stored")
        normalised: FloatMatrix = (matrix / norms).astype(np.float32)
        return normalised

    # ------------------------------------------------------------------ backend hooks
    @abstractmethod
    def _backend_upsert(self, items: list[VectorItem], matrix: FloatMatrix) -> None:
        """Store normalised rows of ``matrix`` under ``items[i].id`` (insert or replace)."""

    @abstractmethod
    def _backend_delete(self, ids: list[str]) -> None:
        """Remove IDs that are known to be present."""

    @abstractmethod
    def _backend_search(
        self, query: FloatMatrix, k: int, source: str | None
    ) -> list[tuple[str, float]]:
        """Return ``(id, cosine score)`` pairs, best first; the store is non-empty and, when
        ``source`` is given, at least one item has that source."""

    @abstractmethod
    def _backend_params(self) -> dict[str, Any]:
        """Backend-specific index parameters merged into :meth:`index_params`."""


def _model_label(model: tuple[str, str | None]) -> str:
    model_id, version = model
    return f"{model_id}@{version}" if version else model_id


def finite_score(value: float) -> float:
    """Convert a backend score to a plain float, rejecting NaN/inf from corrupt indexes."""
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"backend returned a non-finite score: {score}")
    return score


__all__ = [
    "EMBEDDED_SETUP_SECTION",
    "BaseVectorStore",
    "FloatMatrix",
    "HealthStatus",
    "Hit",
    "Vector",
    "VectorItem",
    "VectorStore",
    "finite_score",
]
