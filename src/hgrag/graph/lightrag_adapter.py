"""LightRAG adapter: our chunks in (D2), structured context out (D1), every LLM call metered (D3).

* Indexing: each document is inserted with ``ids = file_paths = doc_id`` and a custom
  ``chunking_func`` that returns *our* precomputed chunks for that document's text, so the graph
  and vector indexes share chunks exactly (checked by a test).
* Querying: ``aquery_data`` returns entities, relations and chunks without generating an answer;
  our pipelines generate with the shared prompt.
* LightRAG's "extract" role (entity/relation extraction, description summaries) goes to our
  ``extractor`` role; its "keyword" role goes to ``keywords``. All through :class:`LLMGateway`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hgrag.config import AppConfig
from hgrag.core.hashing import stable_hash
from hgrag.index.chunking import Chunk
from hgrag.index.embedders import Embedder
from hgrag.index.tokens import count_tokens
from hgrag.llm.gateway import LLMGateway
from hgrag.llm.types import Message
from hgrag.pipelines.base import ContextUnit

EXTRACT_ROLE = "extractor"
KEYWORD_ROLE = "keywords"
GRAPH_MODES = ("naive", "local", "global", "hybrid", "mix")
_SEP = "<SEP>"
# kwargs LightRAG passes to its LLM function that our gateway handles itself (or must not forward).
_DROPPED_KWARGS = {"response_format", "keyword_extraction", "hashing_kv", "stream", "enable_cot"}
_JSON_HINT = "Reply with a single JSON object only, with no surrounding text."


@dataclass(frozen=True)
class GraphLocation:
    """One graph = one LightRAG *workspace* under the shared root ``data/graphs/``.

    LightRAG keeps process-wide namespaces per workspace, so distinct workspaces let several
    graphs (datasets, extractors, backends) be open in one process; on server backends
    (Neo4j, Postgres) the workspace also isolates the data. The embedder is part of the
    identity because entity/relation vectors depend on it.
    """

    dataset: str
    chunker: str
    extractor: str
    embedder: str
    backend: str
    sample: bool = False

    @property
    def workspace(self) -> str:
        extractor = f"{self.extractor}-sample" if self.sample else self.extractor
        return "__".join([self.dataset, self.chunker, extractor, self.embedder, self.backend])

    @staticmethod
    def root(config: AppConfig) -> Path:
        return config.paths.data_dir / "graphs"

    def working_dir(self, config: AppConfig) -> Path:
        """Where this graph's files live (LightRAG: ``root / workspace``)."""
        return self.root(config) / self.workspace

    @property
    def run_id(self) -> str:
        """Ledger run ID for indexing spend (per-run budget cap applies to a whole build)."""
        tag = "sample" if self.sample else "full"
        return f"graph.{self.dataset}.{self.chunker}.{self.extractor}.{self.embedder}.{tag}"


def _messages(prompt: str, history: Sequence[dict[str, str]] | None) -> list[Message]:
    turns = [
        Message("assistant" if m.get("role") == "assistant" else "user", str(m.get("content", "")))
        for m in history or []
    ]
    return [*turns, Message("user", prompt)]


def make_llm_func(gateway: LLMGateway, role: str, model_key: str | None, max_tokens: int) -> Any:
    """A LightRAG-compatible async LLM function that routes through our gateway."""

    async def llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        wants_json = bool(kwargs.get("response_format")) or bool(kwargs.get("keyword_extraction"))
        system = system_prompt or ""
        if wants_json:  # Claude has no response_format; ask for JSON in the system prompt
            system = f"{system}\n\n{_JSON_HINT}".strip()
        response = await gateway.acomplete(
            role,
            _messages(prompt, history_messages),
            system=system or None,
            max_tokens=max_tokens,
            model_key=model_key,
        )
        return response.text

    return llm


def make_embedding_func(embedder: Embedder) -> Any:
    """LightRAG ``EmbeddingFunc`` over a local embedder (one call at a time on the GPU)."""
    from lightrag.utils import EmbeddingFunc

    lock = threading.Lock()
    dim = int(embedder.embed_documents(["dimension probe"]).shape[1])

    def encode(texts: list[str], context: str | None) -> np.ndarray[Any, Any]:
        with lock:
            if context == "query":
                return embedder.embed_queries(texts)
            return embedder.embed_documents(texts)

    async def embed(texts: list[str], context: str | None = None, **_: Any) -> Any:
        return await asyncio.to_thread(encode, list(texts), context)

    return EmbeddingFunc(embedding_dim=dim, func=embed, max_token_size=8192)


@dataclass
class ChunkRegistry:
    """Our chunks per document; LightRAG's chunker callback looks them up by document text."""

    by_text_hash: dict[str, list[Chunk]] = field(default_factory=dict)
    by_doc_and_text: dict[tuple[str, str], str] = field(default_factory=dict)

    def register(self, document_text: str, chunks: Sequence[Chunk]) -> None:
        self.by_text_hash[stable_hash(document_text)] = list(chunks)
        for chunk in chunks:
            self.by_doc_and_text.setdefault((chunk.doc_id, chunk.text), chunk.chunk_id)

    def chunking_func(
        self,
        tokenizer: Any,
        content: str,
        split_by_character: str | None,
        split_by_character_only: bool,
        overlap: int,
        size: int,
    ) -> list[dict[str, Any]]:
        chunks = self.by_text_hash.get(stable_hash(content))
        if chunks is None:
            raise KeyError("LightRAG asked to chunk a text we did not register (not our corpus)")
        return [
            {"tokens": c.n_tokens, "content": c.text, "chunk_order_index": i}
            for i, c in enumerate(chunks)
        ]

    def our_chunk_id(self, doc_id: str, text: str) -> str | None:
        return self.by_doc_and_text.get((doc_id, text))


def build_lightrag(
    config: AppConfig,
    location: GraphLocation,
    *,
    gateway: LLMGateway,
    embedder: Embedder,
    registry: ChunkRegistry,
    extractor_model_key: str | None,
    keyword_model_key: str | None = None,
) -> Any:
    """Construct (not yet initialised) a LightRAG instance for ``location``."""
    from lightrag import LightRAG
    from lightrag.llm_roles import RoleLLMConfig

    graph = config.graph
    backend = graph.backends[location.backend]
    location.working_dir(config).mkdir(parents=True, exist_ok=True)
    extract = make_llm_func(gateway, EXTRACT_ROLE, extractor_model_key, max_tokens=4096)
    keywords = make_llm_func(gateway, KEYWORD_ROLE, keyword_model_key, max_tokens=512)
    return LightRAG(
        working_dir=str(location.root(config)),
        workspace=location.workspace,
        kv_storage=backend.kv_storage,
        vector_storage=backend.vector_storage,
        graph_storage=backend.graph_storage,
        doc_status_storage=backend.doc_status_storage,
        llm_model_func=extract,
        role_llm_configs={
            "extract": RoleLLMConfig(func=extract),
            "keyword": RoleLLMConfig(func=keywords),
        },
        embedding_func=make_embedding_func(embedder),
        chunking_func=registry.chunking_func,
        entity_extract_max_gleaning=graph.extraction.max_gleaning,
        llm_model_max_async=graph.extraction.llm_max_async,
        embedding_func_max_async=graph.extraction.embedding_max_async,
        enable_llm_cache=False,  # our gateway already caches every call by content hash
    )


@dataclass(frozen=True)
class GraphContext:
    """What one graph-mode retrieval returned, converted to our context units."""

    entities: list[ContextUnit]
    relations: list[ContextUnit]
    chunks: list[ContextUnit]
    keywords: dict[str, list[str]]

    def units(self) -> list[ContextUnit]:
        """Order shown to the generator: knowledge-graph facts first, then text (as LightRAG)."""
        return [*self.entities, *self.relations, *self.chunks]


def to_context(data: dict[str, Any], registry: ChunkRegistry) -> GraphContext:
    """Convert ``aquery_data`` output; chunk units get *our* chunk IDs (chunk-level metrics)."""
    payload = data.get("data", {}) or {}
    entities = [
        ContextUnit(
            unit_id=f"ent:{e['entity_name']}",
            kind="entity",
            text=(
                f"{e['entity_name']} ({e.get('entity_type', 'entity')}): {e.get('description', '')}"
            ),
            source_doc_ids=_docs(e.get("file_path", "")),
        )
        for e in payload.get("entities", [])
    ]
    relations = [
        ContextUnit(
            unit_id=f"rel:{r['src_id']}->{r['tgt_id']}",
            kind="relation",
            text=(
                f"{r['src_id']} -> {r['tgt_id']} [{r.get('keywords', '')}]: "
                f"{r.get('description', '')}"
            ),
            score=float(r.get("weight", 0.0) or 0.0),
            source_doc_ids=_docs(r.get("file_path", "")),
        )
        for r in payload.get("relationships", [])
    ]
    chunks = []
    for c in payload.get("chunks", []):
        doc_id = str(c.get("file_path", ""))
        ours = registry.our_chunk_id(doc_id, str(c.get("content", "")))
        chunks.append(
            ContextUnit(
                unit_id=ours or f"lr:{c.get('chunk_id', '')}",
                kind="chunk",
                text=str(c.get("content", "")),
                source_doc_ids=[doc_id] if doc_id else [],
            )
        )
    meta = data.get("metadata", {}) or {}
    keywords = meta.get("keywords", {}) or {}
    return GraphContext(
        entities,
        relations,
        chunks,
        {k: list(v) for k, v in keywords.items() if isinstance(v, list)},
    )


def unit_tokens(unit: ContextUnit) -> int:
    return count_tokens(unit.text)


def _docs(file_path: str) -> list[str]:
    return [p for p in str(file_path).split(_SEP) if p and p != "unknown_source"]
