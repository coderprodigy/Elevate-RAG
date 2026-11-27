"""Indexed lexical and dense retrieval over a shared chunk catalog."""

import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Settings
from .data import (
    Chunk,
    Tokenizer,
    chunk_document,
    documents_at,
    read_jsonl,
    stable_hash,
    write_jsonl,
)
from .providers import Providers, Vector, normalized_vectors, terms


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    score: float
    rerank_score: float | None = None


def sorted_hits(scores: Iterable[tuple[str, float]], limit: int) -> list[Hit]:
    return [
        Hit(key, float(score)) for key, score in sorted(scores, key=lambda x: (-x[1], x[0]))[:limit]
    ]


class BM25:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.lengths = {}
        for chunk in chunks:
            frequency = Counter(terms(f"{chunk.title} {chunk.text}"))
            self.lengths[chunk.chunk_id] = frequency.total()
            for token, count in frequency.items():
                self.postings[token].append((chunk.chunk_id, count))
        self.avg = sum(self.lengths.values()) / max(len(self.lengths), 1)

    def search(self, query: str, limit: int) -> list[Hit]:
        scores: dict[str, float] = defaultdict(float)
        for term in sorted(set(terms(query))):
            posting = self.postings.get(term, [])
            idf = math.log1p((len(self.lengths) - len(posting) + 0.5) / (len(posting) + 0.5))
            for chunk_id, frequency in posting:
                normalization = 1 - self.b + self.b * self.lengths[chunk_id] / self.avg
                scores[chunk_id] += (
                    idf * frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
                )
        return sorted_hits(scores.items(), limit)


def rrf(rankings: Iterable[Iterable[Hit]], constant: int, limit: int) -> list[Hit]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        unique = dict.fromkeys(hit.chunk_id for hit in ranking)
        for rank, chunk_id in enumerate(unique, 1):
            scores[chunk_id] += 1 / (constant + rank)
    return sorted_hits(scores.items(), limit)


class VectorIndex(Protocol):
    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None: ...

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]: ...

    def size(self) -> int: ...

    def close(self) -> None: ...


class ExactIndex:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.ids: list[str] = []
        self.positions: dict[str, int] = {}
        self.sources: list[str] = []
        self.vectors = np.empty((0, dimensions), np.float32)

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None:
        matrix = normalized_vectors(vectors, len(chunks), self.dimensions)
        overlap = set(self.ids).intersection(chunk.chunk_id for chunk in chunks)
        if overlap:
            raise ValueError("in-process index cannot update existing rows")
        for chunk in chunks:
            self.positions[chunk.chunk_id] = len(self.ids)
            self.ids.append(chunk.chunk_id)
        self.sources.extend(chunk.source for chunk in chunks)
        self.vectors = np.vstack((self.vectors, matrix))

    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]:
        checked = normalized_vectors([query], 1, self.dimensions)[0]
        return sorted_hits(
            (
                (key, score)
                for key, score, src in zip(
                    self.ids, self.vectors @ checked, self.sources, strict=True
                )
                if source is None or src == source
            ),
            limit,
        )

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        return {
            key: self.vectors[self.positions[key]].tolist() for key in ids if key in self.positions
        }

    def size(self) -> int:
        return len(self.ids)

    def close(self) -> None:
        pass


class FaissIndex:
    def __init__(
        self,
        dimensions: int,
        *,
        m: int | None = None,
        construction: int = 200,
        search_ef: int = 100,
    ):
        import faiss

        self.dimensions = dimensions
        self.ids: list[str] = []
        self.positions: dict[str, int] = {}
        self.sources: list[str] = []
        self.faiss = faiss
        self.index: Any = (
            faiss.IndexFlatIP(dimensions)
            if m is None
            else faiss.IndexHNSWFlat(dimensions, m, faiss.METRIC_INNER_PRODUCT)
        )
        self.search_ef = search_ef
        if m is not None:
            hnsw = getattr(self.index, "hnsw")
            hnsw.efConstruction = construction
            hnsw.efSearch = search_ef

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None:
        matrix = normalized_vectors(vectors, len(chunks), self.dimensions)
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids) or set(self.ids).intersection(ids):
            raise ValueError("FAISS append requires new, unique chunk IDs")
        self.index.add(matrix)
        self.positions.update({key: i for i, key in enumerate(ids, start=len(self.ids))})
        self.ids.extend(ids)
        self.sources.extend(chunk.source for chunk in chunks)

    def save(self, path: Path) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            pending = Path(handle.name)
        try:
            self.faiss.write_index(self.index, str(pending))
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def restore(self, path: Path, chunks: Sequence[Chunk]) -> None:
        loaded = self.faiss.read_index(str(path))
        if (
            loaded.d != self.dimensions
            or loaded.ntotal != len(chunks)
            or type(loaded) is not type(self.index)
        ):
            raise ValueError(
                "persisted FAISS index does not match the corpus and embedding contract"
            )
        self.index = loaded
        self.ids = [chunk.chunk_id for chunk in chunks]
        self.positions = {key: i for i, key in enumerate(self.ids)}
        self.sources = [chunk.source for chunk in chunks]

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        return {
            key: self.index.reconstruct(self.positions[key]).tolist()
            for key in ids
            if key in self.positions
        }

    def size(self) -> int:
        return len(self.ids)

    def close(self) -> None:
        pass

    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]:
        checked = normalized_vectors([query], 1, self.dimensions)
        if not self.ids or limit < 1:
            return []
        search_limit = min(limit, len(self.ids))
        params: Any = None
        if hasattr(self.index, "hnsw"):
            params = self.faiss.SearchParametersHNSW()
            params.efSearch = ef or self.search_ef
        if source is not None:
            matching = np.asarray(
                [i for i, value in enumerate(self.sources) if value == source], dtype=np.int64
            )
            if not len(matching):
                return []
            selector = self.faiss.IDSelectorBatch(matching)
            params = params or self.faiss.SearchParameters()
            params.sel = selector
        distances, indexes = self.index.search(checked, search_limit, params=params)
        return sorted_hits(
            (
                (self.ids[index], score)
                for index, score in zip(indexes[0], distances[0], strict=True)
                if index >= 0
            ),
            limit,
        )


def corpus_key(manifest: Mapping[str, Any], settings: Settings, tokenizer: Tokenizer) -> str:
    return stable_hash(
        "chunks-v3",
        manifest["dataset"],
        manifest["version"],
        manifest["files"]["documents.jsonl"],
        tokenizer.identity,
        settings.chunk_size,
        settings.overlap,
        settings.paragraph_chunks,
    )


def load_chunks(
    dataset: Path, index: Path, settings: Settings, tokenizer: Tokenizer
) -> dict[str, Chunk]:
    path = index / "chunks.jsonl"
    if not path.exists():
        chunks = (
            chunk
            for document in documents_at(dataset)
            for chunk in chunk_document(
                document,
                tokenizer,
                settings.chunk_size,
                settings.overlap,
                settings.paragraph_chunks,
            )
        )
        write_jsonl(path, map(asdict, chunks))
    catalog = {row["chunk_id"]: Chunk(**row) for row in read_jsonl(path)}
    if not catalog:
        raise ValueError("cannot index an empty chunk catalog")
    return catalog


def vector_key(settings: Settings, chunks_key: str) -> str:
    return stable_hash(
        "vectors-v1",
        chunks_key,
        asdict(settings.model("embedding")),
        settings.vector_store,
        settings.namespace,
        settings.hnsw_m,
        settings.ef_construction,
        settings.ef_search,
    )


async def build_vectors(providers: Providers, chunks: Sequence[Chunk], path: Path) -> Vector:
    dimensions = providers.settings.model("embedding").dimensions
    if path.exists():
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (len(chunks), dimensions) or matrix.dtype != np.float32:
            raise ValueError(f"incompatible persisted vector index at {path}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"nonfinite persisted vector index at {path}")
        return matrix
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as temp:
        pending = Path(temp.name)
    try:
        matrix = np.lib.format.open_memmap(
            pending,
            mode="w+",
            dtype=np.float32,
            shape=(len(chunks), dimensions),
        )
        batch = providers.settings.embedding_batch
        for start in range(0, len(chunks), batch):
            text = [chunk.text for chunk in chunks[start : start + batch]]
            matrix[start : start + len(text)] = await providers.embed(text)
        matrix.flush()
        os.replace(pending, path)
        return np.load(path, mmap_mode="r", allow_pickle=False)
    finally:
        pending.unlink(missing_ok=True)
