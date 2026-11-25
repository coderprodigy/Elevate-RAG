"""Query-time retrieval over a built index: BM25 | dense | hybrid (RRF), optional reranking,
then ``final_k`` and the shared token budget. Results are cached by content hash.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

from hgrag.config import AppConfig, SearchSettings
from hgrag.core.hashing import stable_hash
from hgrag.index.bm25_index import BM25Index
from hgrag.index.build import load_chunks, load_manifest
from hgrag.index.chunking import Chunk
from hgrag.index.embedders import Embedder
from hgrag.index.paths import IndexPaths
from hgrag.index.vector_index import FaissIndex
from hgrag.llm.cache import ResponseCache
from hgrag.retrieval.budget import apply_token_budget
from hgrag.retrieval.fusion import reciprocal_rank_fusion
from hgrag.retrieval.rerank import Reranker

RetrieverKind = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class RetrievalSpec:
    retriever: RetrieverKind
    rerank: bool


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Searcher:
    """Thread-safe (one lock around model use), so pipelines can call it via ``to_thread``."""

    def __init__(
        self,
        config: AppConfig,
        dataset: str,
        chunker: str,
        spec: RetrievalSpec,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        settings: SearchSettings | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        if spec.retriever in ("dense", "hybrid") and embedder is None:
            raise ValueError(f"{spec.retriever} retrieval needs an embedder")
        if spec.rerank and reranker is None:
            raise ValueError("reranking needs a reranker")
        self.dataset, self.chunker, self.spec = dataset, chunker, spec
        self.settings = settings or config.retrieval.search
        self._paths = IndexPaths.of(config, dataset, chunker)
        self._manifest = load_manifest(self._paths)
        self._chunks = {c.chunk_id: c for c in load_chunks(self._paths)}
        ids = list(self._chunks)
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = BM25Index.load(self._paths.bm25, ids) if spec.retriever != "dense" else None
        self._dense = (
            FaissIndex.load(self._paths.hnsw(embedder.name), ids, config.retrieval.hnsw.ef_search)
            if embedder is not None and spec.retriever != "bm25"
            else None
        )
        self._cache = cache or ResponseCache(config.paths.runs_dir / "cache" / "retrieval.sqlite")
        self._lock = threading.Lock()

    def describe(self) -> dict[str, object]:
        return {
            "index": {"dataset": self.dataset, "chunker": self.chunker},
            "chunk_set_hash": self._manifest.chunk_set_hash,
            "retriever": self.spec.retriever,
            "embedder": self._embedder.name if self._embedder and self._dense else None,
            "reranker": self._reranker.name if self.spec.rerank and self._reranker else None,
            "search": self.settings.model_dump(),
        }

    def cached(self, question: str) -> list[Hit] | None:
        """Hits from the retrieval cache without computing anything (None if not cached)."""
        stored = self._cache.get(self._key(question))
        return None if stored is None else self._finish(stored)

    def _key(self, question: str) -> str:
        return stable_hash({"retrieval": self.describe(), "question": question})

    def _finish(self, stored: dict[str, Any]) -> list[Hit]:
        hits = [Hit(self._chunks[cid], score) for cid, score in stored["hits"]]
        return apply_token_budget(
            hits, [h.chunk.n_tokens for h in hits], self.settings.context_token_budget
        )

    def search(self, question: str) -> list[Hit]:
        key = self._key(question)
        cached = self._cache.get(key)
        if cached is None:
            with self._lock:
                ranked = self._rank(question)
            cached = {"hits": ranked}
            self._cache.put(key, cached)
        return self._finish(cached)

    def _rank(self, question: str) -> list[tuple[str, float]]:
        n = self.settings.candidates
        lists: list[list[tuple[str, float]]] = []
        if self._bm25 is not None:
            lists.append(self._bm25.search([question], n)[0])
        if self._dense is not None and self._embedder is not None:
            lists.append(self._dense.search(self._embedder.embed_queries([question]), n)[0])

        if len(lists) == 1:
            candidates = lists[0][:n]
        else:
            fused = reciprocal_rank_fusion(
                [[cid for cid, _ in ranking] for ranking in lists], k=self.settings.rrf_k, limit=n
            )
            candidates = fused

        if self.spec.rerank and self._reranker is not None and candidates:
            # Rerank the best `rerank_candidates`; the rest keep their fused order behind them.
            head = candidates[: self.settings.rerank_candidates]
            tail = candidates[self.settings.rerank_candidates :]
            texts = [self._chunks[cid].indexed_text for cid, _ in head]
            scores = self._reranker.score(question, texts)
            reranked = sorted(
                zip((cid for cid, _ in head), scores, strict=True), key=lambda p: -p[1]
            )
            candidates = reranked + [(cid, float("-inf")) for cid, _ in tail]
        return [(cid, float(score)) for cid, score in candidates[: self.settings.final_k]]
