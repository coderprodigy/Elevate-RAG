"""Query-time access to a built graph: one LightRAG instance per process, results cached.

The LightRAG instance is created lazily inside the running event loop (it binds to its loop).
Keyword-extraction calls made during retrieval are attributed to the query that caused them via
:data:`RETRIEVAL_SPEND`, so graph pipelines report their full per-query cost.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from hgrag.config import AppConfig
from hgrag.core.hashing import stable_hash
from hgrag.graph.build import load_graph_manifest
from hgrag.graph.lightrag_adapter import (
    ChunkRegistry,
    GraphContext,
    GraphLocation,
    build_lightrag,
    to_context,
)
from hgrag.index.build import load_chunks
from hgrag.index.chunking import Chunk
from hgrag.index.embedders import Embedder
from hgrag.index.paths import IndexPaths
from hgrag.llm.cache import ResponseCache
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import ContextUnit


@dataclass
class SpendTally:
    """LLM spend during one query's retrieval (keyword extraction)."""

    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: list[str] = field(default_factory=list)


RETRIEVAL_SPEND: ContextVar[SpendTally | None] = ContextVar("RETRIEVAL_SPEND", default=None)


class GraphIndex:
    def __init__(
        self,
        config: AppConfig,
        location: GraphLocation,
        *,
        gateway: LLMGateway,
        embedder: Embedder,
        keyword_model_key: str | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.location = location
        self.manifest = load_graph_manifest(config, location)
        self._config = config
        self._gateway = gateway
        self._embedder = embedder
        self._keyword_model_key = keyword_model_key
        chunks = load_chunks(IndexPaths.of(config, location.dataset, location.chunker))
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._registry = ChunkRegistry()
        for chunk in chunks:
            self._registry.by_doc_and_text.setdefault((chunk.doc_id, chunk.text), chunk.chunk_id)
        self._cache = cache or ResponseCache(config.paths.runs_dir / "cache" / "retrieval.sqlite")
        self._rag: Any = None
        self._lock = asyncio.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "index": {"dataset": self.location.dataset, "chunker": self.location.chunker},
            "graph": {
                "extractor": self.location.extractor,
                "embedder": self._embedder.name,
                "backend": self.location.backend,
                "chunk_set_hash": self.manifest.chunk_set_hash,
            },
            "graph_query": self._config.graph.query.model_dump(),
        }

    def cache_key(self, question: str, mode: str, params: dict[str, Any]) -> str:
        return stable_hash(
            {"graph": self.describe(), "mode": mode, "params": params, "q": question}
        )

    def cached(self, question: str, mode: str, params: dict[str, Any]) -> GraphContext | None:
        stored = self._cache.get(self.cache_key(question, mode, params))
        return _from_cache(stored) if stored is not None else None

    async def retrieve(self, question: str, mode: str, params: dict[str, Any]) -> GraphContext:
        key = self.cache_key(question, mode, params)
        stored = self._cache.get(key)
        if stored is not None:
            return _from_cache(stored)
        from lightrag import QueryParam

        rag = await self._ensure_rag()
        data = await rag.aquery_data(question, QueryParam(mode=mode, **params))
        if data.get("status") not in (None, "success"):
            message = str(data.get("message", ""))
            if "no results" not in message.lower():
                raise RuntimeError(f"LightRAG query failed: {message}")
            data = {"data": {}, "metadata": data.get("metadata", {})}  # nothing matched: empty
        context = to_context(data, self._registry)
        self._cache.put(key, _to_cache(context))
        return context

    async def aclose(self) -> None:
        """Release LightRAG storages (it holds process-wide namespaces until finalised)."""
        async with self._lock:
            if self._rag is not None:
                await self._rag.finalize_storages()
                self._rag = None

    async def _ensure_rag(self) -> Any:
        async with self._lock:
            if self._rag is None:
                rag = build_lightrag(
                    self._config,
                    self.location,
                    gateway=_TallyingGateway(self._gateway),  # type: ignore[arg-type]
                    embedder=self._embedder,
                    registry=self._registry,
                    extractor_model_key=self.location.extractor,
                    keyword_model_key=self._keyword_model_key,
                )
                await rag.initialize_storages()
                self._rag = rag
            return self._rag


class _TallyingGateway:
    """Gateway proxy that adds each call's spend to the current query's :class:`SpendTally`."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def acomplete(self, role: str, prompt: Any, **kwargs: Any) -> Any:
        response = await self._gateway.acomplete(role, prompt, **kwargs)
        tally = RETRIEVAL_SPEND.get()
        if tally is not None:
            tally.cost_usd += response.cost_usd
            tally.input_tokens += response.input_tokens + response.cached_input_tokens
            tally.output_tokens += response.output_tokens
            tally.calls.append(role)
        return response


def _to_cache(context: GraphContext) -> dict[str, Any]:
    return {
        "entities": [u.model_dump() for u in context.entities],
        "relations": [u.model_dump() for u in context.relations],
        "chunks": [u.model_dump() for u in context.chunks],
        "keywords": context.keywords,
    }


def _from_cache(stored: dict[str, Any]) -> GraphContext:
    return GraphContext(
        entities=[ContextUnit(**u) for u in stored["entities"]],
        relations=[ContextUnit(**u) for u in stored["relations"]],
        chunks=[ContextUnit(**u) for u in stored["chunks"]],
        keywords=stored.get("keywords", {}),
    )
