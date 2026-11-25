"""On-disk layout of an index: ``data/indexes/<dataset>/<chunker>/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hgrag.config import AppConfig


@dataclass(frozen=True)
class IndexPaths:
    root: Path

    @classmethod
    def of(cls, config: AppConfig, dataset: str, chunker: str) -> IndexPaths:
        return cls(config.paths.data_dir / "indexes" / dataset / chunker)

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def gold_chunks(self) -> Path:
        return self.root / "gold_chunks.json"

    @property
    def bm25(self) -> Path:
        return self.root / "bm25"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def embedding_dir(self, embedder: str) -> Path:
        return self.root / f"emb-{embedder}"

    def embeddings(self, embedder: str) -> Path:
        return self.embedding_dir(embedder) / "embeddings.npy"

    def hnsw(self, embedder: str) -> Path:
        return self.embedding_dir(embedder) / "hnsw.faiss"
