"""Expose the immutable corpus vector index through LightRAG's passage contract."""

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from lightrag.base import BaseVectorStorage
from lightrag.utils import EmbeddingFunc, sanitize_text_for_encoding
from lightrag.utils_pipeline import make_custom_chunk_id

from .data import Chunk
from .retrieval import VectorIndex


def native_passage_id(chunk: Chunk) -> str:
    if not chunk.text or sanitize_text_for_encoding(chunk.text) != chunk.text:
        raise ValueError(
            f"{chunk.chunk_id}: shared passage must be normalized for LightRAG ingestion"
        )
    return make_custom_chunk_id(chunk.doc_id, chunk.text)


class SharedPassages(BaseVectorStorage):
    """LightRAG owns graph records. The corpus owner alone writes passage vectors."""

    def __init__(
        self, vector: VectorIndex, chunks: Sequence[Chunk], workspace: str, embedding: EmbeddingFunc
    ) -> None:
        super().__init__(
            namespace="chunks",
            workspace=workspace,
            global_config={},
            embedding_func=embedding,
            cosine_better_than_threshold=-1.0,
        )
        self.vector = vector
        self.native_ids: dict[str, str] = {}
        self.passages: dict[str, Chunk] = {}
        aliases: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            native_id = native_passage_id(chunk)
            self.native_ids[chunk.chunk_id] = native_id
            self.passages.setdefault(native_id, chunk)
            aliases[native_id] += 1
        self.max_aliases = max(aliases.values(), default=1)

    @staticmethod
    def record(key: str, chunk: Chunk) -> dict[str, Any]:
        return {
            "id": key,
            "content": chunk.text,
            "file_path": chunk.source,
            "full_doc_id": chunk.doc_id,
        }

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            return []
        if query_embedding is None:
            assert self.embedding_func is not None
            query_embedding = (await self.embedding_func([query], context="query"))[0].tolist()
        hits = await asyncio.to_thread(
            self.vector.search,
            np.asarray(query_embedding, dtype=np.float32),
            top_k * self.max_aliases,
        )
        records: dict[str, dict[str, Any]] = {}
        for hit in hits:
            key = self.native_ids[hit.chunk_id]
            records.setdefault(key, {**self.record(key, self.passages[key]), "distance": hit.score})
            if len(records) == top_k:
                break
        return list(records.values())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for key, row in data.items():
            if key not in self.passages or self.passages[key].text != row.get("content"):
                raise ValueError(f"graph ingestion attempted to change shared passage {key}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        chunk = self.passages.get(id)
        return self.record(id, chunk) if chunk is not None else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.record(key, self.passages[key]) for key in ids if key in self.passages]

    async def get_by_ids_strict(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [await self.get_by_id(key) for key in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        rows = {key: self.passages[key].chunk_id for key in ids if key in self.passages}
        vectors = await asyncio.to_thread(self.vector.get_vectors, list(rows.values()))
        if set(vectors) != set(rows.values()):
            raise RuntimeError("shared index is missing vectors for existing corpus passages")
        return {key: vectors[chunk_id] for key, chunk_id in rows.items()}

    async def get_exact_count(self) -> int:
        return len(self.passages)

    async def is_empty(self) -> bool:
        return not self.passages

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        # LightRAG can roll back its graph records without mutating the immutable corpus.
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError("shared passage storage has no entities")

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError("shared passage storage has no relationships")

    async def drop(self) -> dict[str, str]:
        return {"status": "error", "message": "the immutable corpus owns this shared index"}
