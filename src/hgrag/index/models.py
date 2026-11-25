"""Load local embedding / reranking models once per process (they take seconds to load)."""

from __future__ import annotations

from functools import lru_cache

from hgrag.config import EmbedderSpec, RerankerSpec
from hgrag.index.embedders import Embedder, SentenceTransformerEmbedder
from hgrag.retrieval.rerank import CrossEncoderReranker, Reranker


@lru_cache(maxsize=4)
def load_embedder(name: str, spec: EmbedderSpec) -> Embedder:
    return SentenceTransformerEmbedder(name, spec)


@lru_cache(maxsize=2)
def load_reranker(name: str, spec: RerankerSpec) -> Reranker:
    return CrossEncoderReranker(name, spec)
