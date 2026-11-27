"""The shared chunk store: the single source of chunk identity for every pipeline (D-02)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from elevate_rag.types import Chunk
from elevate_rag.utils import DataError, read_jsonl, write_jsonl


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: dict[str, Chunk] = {}
        for c in chunks:
            if c.chunk_id in self._chunks:
                raise DataError(f"duplicate chunk id {c.chunk_id}")
            self._chunks[c.chunk_id] = c
        self._by_doc: dict[str, list[str]] = {}
        for c in self._chunks.values():
            self._by_doc.setdefault(c.doc_id, []).append(c.chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def __getitem__(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    @property
    def ids(self) -> list[str]:
        return list(self._chunks)

    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        return [self._chunks[c] for c in self._by_doc.get(doc_id, [])]

    def doc_ids(self) -> list[str]:
        return list(self._by_doc)

    def save(self, path: Path) -> None:
        write_jsonl(path, self._chunks.values())

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        if not path.exists():
            raise DataError(f"chunk store missing: {path}")
        return cls(Chunk(**r) for r in read_jsonl(path))
