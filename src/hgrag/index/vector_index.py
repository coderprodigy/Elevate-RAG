"""FAISS vector index (HNSW with fixed M / efConstruction / efSearch, or exact flat search).

FAISS is the Phase 3 store; Qdrant, Milvus and pgvector adapters arrive with the database
comparison (Phase 6) behind the same ``search`` signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from hgrag.config import HNSWSpec
from hgrag.index.embedders import Vectors


class FaissIndex:
    """Maps FAISS row numbers back to chunk IDs; scores are cosine similarities."""

    def __init__(self, index: Any, ids: list[str]) -> None:
        self._index = index
        self._ids = ids

    @classmethod
    def build(cls, vectors: Vectors, ids: list[str], hnsw: HNSWSpec | None) -> FaissIndex:
        dim = vectors.shape[1]
        if hnsw is None:
            index: Any = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexHNSWFlat(dim, hnsw.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = hnsw.ef_construction
            index.hnsw.efSearch = hnsw.ef_search
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index, ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path, ids: list[str], ef_search: int | None = None) -> FaissIndex:
        index = faiss.read_index(str(path))
        if ef_search is not None and hasattr(index, "hnsw"):
            index.hnsw.efSearch = ef_search
        return cls(index, ids)

    def search(self, query_vectors: Vectors, k: int) -> list[list[tuple[str, float]]]:
        """Top ``k`` (chunk_id, score) per query vector, best first."""
        scores, rows = self._index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)
        return [
            [(self._ids[r], float(s)) for r, s in zip(row, score, strict=True) if r != -1]
            for row, score in zip(rows, scores, strict=True)
        ]
