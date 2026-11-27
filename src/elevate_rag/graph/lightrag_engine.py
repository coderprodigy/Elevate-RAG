"""Thin adapter around LightRAG 1.5.7 (D-02, D-03).

* Indexing: our chunks go in unchanged via `ainsert_custom_chunks`; LightRAG chunk IDs are
  recomputed with its public `make_custom_chunk_id` so every entity/relation's `source_id` maps
  back to an Elevate-RAG chunk ID (FR-G1 provenance).
* Retrieval: `aquery_data` returns the entities / relations / chunks of a mode without generation;
  they become typed :class:`ContextItem`s for the shared reranker + budget + prompt.
* All LLM / embedding calls go through Elevate-RAG providers, so they are cached, budgeted and
  recorded in the ledger. LightRAG's own LLM cache is disabled to keep one source of truth.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from elevate_rag.config import PipelineConfig
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.providers.embeddings import BaseEmbedder, Kind
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.types import Chunk, ContextItem, Document, RetrievalResult
from elevate_rag.utils import ElevateError, stable_hash, write_json

log = logging.getLogger(__name__)

GRAPH_MODES = ("naive", "local", "global", "hybrid", "mix")
SEP = "<SEP>"


def _llm_func(llm: BaseLLM, task: str, max_tokens: int) -> Any:
    async def func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        msgs = [m for m in (history_messages or []) if m.get("role") in ("user", "assistant")]
        msgs.append({"role": "user", "content": prompt})
        res = await llm.complete(
            msgs,
            system=system_prompt,
            task=task,
            max_tokens=int(kwargs.get("max_tokens") or max_tokens),
        )
        return res.text

    return func


class LightRAGEngine:
    def __init__(
        self,
        cfg: PipelineConfig,
        workdir: Path,
        *,
        embedder: BaseEmbedder,
        extractor: BaseLLM | None,
        keyword_llm: BaseLLM | None,
        query_llm: BaseLLM,
    ) -> None:
        self.cfg = cfg
        self.workdir = workdir
        self.embedder = embedder
        self.extractor = extractor
        self.keyword_llm = keyword_llm or extractor
        self.query_llm = query_llm
        self.rag: Any = None
        self._lr_to_ours: dict[str, str] = {}

    # ------------------------------------------------------------------------- lifecycle
    async def open(self) -> LightRAGEngine:
        from lightrag import LightRAG
        from lightrag.llm_roles import RoleLLMConfig
        from lightrag.utils import EmbeddingFunc

        # lightrag.utils forces its logger to INFO on import; keep it quiet unless asked.
        logging.getLogger("lightrag").setLevel(os.getenv("ELEVATE_LIGHTRAG_LOG_LEVEL", "WARNING"))

        lc = self.cfg.lightrag
        emb = self.embedder

        async def embed(texts: list[str], context: str = "document", **_: Any) -> np.ndarray:
            import asyncio

            kind: Kind = "query" if context == "query" else "document"
            return await asyncio.to_thread(emb.embed, list(texts), kind)

        base_llm = self.extractor or self.query_llm
        roles = {"query": RoleLLMConfig(func=_llm_func(self.query_llm, "lightrag:query", 2048))}
        if self.extractor is not None:
            roles["extract"] = RoleLLMConfig(
                func=_llm_func(
                    self.extractor, "lightrag:extract", self.cfg.lightrag.extract_max_tokens
                )
            )
        if self.keyword_llm is not None:
            roles["keyword"] = RoleLLMConfig(
                func=_llm_func(self.keyword_llm, "lightrag:keywords", 512)
            )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rag = LightRAG(
            working_dir=str(self.workdir),
            # LightRAG keeps process-wide shared state keyed by `workspace:namespace`; a unique
            # workspace per physical index keeps indexes in one process isolated
            # (regression-tested).
            workspace=f"elevate_{stable_hash(str(self.workdir.resolve()), 16)}",
            kv_storage=lc.kv_storage,
            vector_storage=lc.vector_storage,
            graph_storage=lc.graph_storage,
            doc_status_storage=lc.doc_status_storage,
            embedding_func=EmbeddingFunc(
                embedding_dim=emb.dimension,
                func=embed,
                max_token_size=8192,
                model_name=emb.model_id,
                supports_asymmetric=True,
            ),
            llm_model_func=_llm_func(base_llm, "lightrag:summary", 1024),
            llm_model_name=base_llm.cfg.model,
            role_llm_configs=roles,
            chunk_token_size=self.cfg.chunking.chunk_tokens + 256,
            chunk_overlap_token_size=0,
            entity_extract_max_gleaning=lc.entity_extract_max_gleaning,
            force_llm_summary_on_merge=lc.force_llm_summary_on_merge,
            llm_model_max_async=lc.llm_max_async,
            max_parallel_insert=lc.max_parallel_insert,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": lc.cosine_better_than_threshold,
                **lc.vector_storage_kwargs,
            },
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
            top_k=lc.top_k,
            chunk_top_k=lc.chunk_top_k,
        )
        await self.rag.initialize_storages()
        return self

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None

    # ------------------------------------------------------------------------- indexing
    @staticmethod
    def lightrag_chunk_id(chunk: Chunk) -> str:
        from lightrag.utils_pipeline import make_custom_chunk_id

        return str(make_custom_chunk_id(chunk.doc_id, chunk.text))

    def set_chunk_map(self, store: ChunkStore) -> None:
        self._lr_to_ours = {self.lightrag_chunk_id(c): c.chunk_id for c in store.chunks()}

    async def index(
        self, docs: Sequence[Document], store: ChunkStore, progress_path: Path
    ) -> dict[str, Any]:
        """Insert each document's chunks. Resumable: finished doc IDs are appended to
        `progress_path`; failures are collected and reported, never swallowed."""
        if self.extractor is None:
            raise ElevateError("graph indexing needs models.extractor")
        done: set[str] = set()
        if progress_path.exists():
            done = {ln.strip() for ln in progress_path.read_text().splitlines() if ln.strip()}
        failures: dict[str, str] = {}
        t0 = time.perf_counter()
        for i, doc in enumerate(docs):
            if doc.doc_id in done:
                continue
            chunks = store.doc_chunks(doc.doc_id)
            if not chunks:
                continue
            try:
                await self.rag.ainsert_custom_chunks(
                    doc.text, [c.text for c in chunks], doc_id=doc.doc_id
                )
            except Exception as exc:  # recorded and re-raised as a summary below
                failures[doc.doc_id] = f"{type(exc).__name__}: {exc}"[:500]
                log.error("graph indexing failed for %s: %s", doc.doc_id, exc)
                continue
            with progress_path.open("a") as fh:
                fh.write(doc.doc_id + "\n")
            if (i + 1) % 50 == 0:
                log.info("graph-indexed %d/%d documents", i + 1, len(docs))
        self.set_chunk_map(store)
        await self.verify_chunk_identity(store)
        return {
            "documents": len(docs),
            "failed": failures,
            "seconds": round(time.perf_counter() - t0, 3),
        }

    async def verify_chunk_identity(self, store: ChunkStore, sample: int = 200) -> None:
        """Regression guard for D-02: LightRAG must hold exactly our chunk texts."""
        ids = list(self._lr_to_ours)[:sample]
        rows = await self.rag.text_chunks.get_by_ids(ids)
        for lr_id, row in zip(ids, rows, strict=True):
            if row is None:
                continue  # document may have failed; reported by index()
            ours = store[self._lr_to_ours[lr_id]]
            if row.get("content") != ours.text or row.get("full_doc_id") != ours.doc_id:
                raise ElevateError(
                    f"chunk identity broken for {ours.chunk_id}: LightRAG stored "
                    "different content — check the LightRAG version (D-02)"
                )

    # ------------------------------------------------------------------------- retrieval
    def _map_sources(self, source_id: str | None) -> list[str]:
        out = []
        for lr in (source_id or "").split(SEP):
            if lr and (ours := self._lr_to_ours.get(lr.strip())):
                out.append(ours)
        return list(dict.fromkeys(out))

    async def retrieve(self, query: str, mode: str, store: ChunkStore) -> RetrievalResult:
        from lightrag import QueryParam

        if mode not in GRAPH_MODES:
            raise ElevateError(f"unknown LightRAG mode {mode!r}")
        rc, lc = self.cfg.retrieval, self.cfg.lightrag
        kg_tokens = int(rc.context_budget * rc.kg_share)
        param = QueryParam(
            mode=mode,
            top_k=lc.top_k,
            chunk_top_k=lc.chunk_top_k,
            max_entity_tokens=max(1, kg_tokens // 2),
            max_relation_tokens=max(1, kg_tokens // 2),
            max_total_tokens=rc.context_budget * 2,
            enable_rerank=False,
        )
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        res = await self.rag.aquery_data(query, param)
        timings["graph_retrieval"] = time.perf_counter() - t0
        if res.get("status") != "success":
            # LightRAG reports "no context found" as a failure status; that is a legitimate
            # empty retrieval, not an error.
            log.info(
                "LightRAG %s returned status=%s: %s", mode, res.get("status"), res.get("message")
            )
        data = res.get("data") or {}
        items: list[ContextItem] = []
        for rank, e in enumerate(data.get("entities", [])):
            items.append(
                ContextItem(
                    item_id=f"entity:{e['entity_name']}",
                    kind="entity",
                    text=(
                        f"{e['entity_name']} ({e.get('entity_type', '?')}): "
                        f"{e.get('description', '')}"
                    ),
                    score=1.0 / (1 + rank),
                    source_chunk_ids=self._map_sources(e.get("source_id")),
                    metadata={"name": e["entity_name"]},
                )
            )
        for rank, r in enumerate(data.get("relationships", [])):
            items.append(
                ContextItem(
                    item_id=f"relation:{r['src_id']}|{r['tgt_id']}",
                    kind="relation",
                    text=f"{r['src_id']} —[{r.get('keywords', '')}]→ {r['tgt_id']}: "
                    f"{r.get('description', '')}",
                    score=1.0 / (1 + rank),
                    source_chunk_ids=self._map_sources(r.get("source_id")),
                    metadata={"weight": r.get("weight")},
                )
            )
        for rank, c in enumerate(data.get("chunks", [])):
            ours = self._lr_to_ours.get(c.get("chunk_id", ""))
            if ours is None:
                log.warning("LightRAG returned unmapped chunk %s", c.get("chunk_id"))
                continue
            ch = store[ours]
            items.append(
                ContextItem(
                    item_id=ours,
                    kind="chunk",
                    text=ch.text,
                    doc_id=ch.doc_id,
                    score=1.0 / (1 + rank),
                )
            )
        if rc.graph_depth == 2 and mode in ("local", "hybrid", "mix"):
            t1 = time.perf_counter()
            items += await self._expand_two_hop(items, rc.max_kg_items)
            timings["two_hop"] = time.perf_counter() - t1
        return RetrievalResult(
            mode=mode,
            items=items,
            timings=timings,
            metadata={
                "keywords": (res.get("metadata") or {}).get("keywords"),
                "status": res.get("status"),
            },
        )

    async def _expand_two_hop(self, items: list[ContextItem], limit: int) -> list[ContextItem]:
        """Add relations one hop beyond the retrieved entities (A-ablation: graph depth)."""
        g = self.rag.chunk_entity_relation_graph
        seeds = [it.metadata["name"] for it in items if it.kind == "entity"]
        have = {it.item_id for it in items}
        out: list[ContextItem] = []
        neighbours: list[str] = []
        for name in seeds:
            for src, tgt in (await g.get_node_edges(name)) or []:
                other = tgt if src == name else src
                if other not in seeds:
                    neighbours.append(other)
        for name in dict.fromkeys(neighbours):
            for src, tgt in (await g.get_node_edges(name)) or []:
                iid = f"relation:{src}|{tgt}"
                if iid in have or len(out) >= limit:
                    continue
                edge = await g.get_edge(src, tgt) or {}
                have.add(iid)
                out.append(
                    ContextItem(
                        item_id=iid,
                        kind="relation",
                        score=0.0,
                        text=f"{src} —[{edge.get('keywords', '')}]→ {tgt}: "
                        f"{edge.get('description', '')}",
                        source_chunk_ids=self._map_sources(edge.get("source_id")),
                        metadata={"hop": 2},
                    )
                )
        return out

    # ------------------------------------------------------------------------- graph access
    async def export_graph(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        g = self.rag.chunk_entity_relation_graph
        nodes = await g.get_all_nodes()
        edges = await g.get_all_edges()
        for n in nodes:
            n["chunk_ids"] = self._map_sources(n.get("source_id"))
        for e in edges:
            e["chunk_ids"] = self._map_sources(e.get("source_id"))
        return nodes, edges

    async def merge_entities(self, sources: list[str], target: str) -> None:
        # LightRAG 1.5.7's defaults are exactly the strategy we want (descriptions concatenated,
        # first entity type kept, source/file ids joined uniquely); passing `merge_strategy`
        # explicitly is deprecated there and only produced a warning per merge.
        await self.rag.amerge_entities(sources, target)

    def save_chunk_map(self, path: Path) -> None:
        write_json(path, self._lr_to_ours)

    def load_chunk_map(self, path: Path) -> None:
        self._lr_to_ours = json.loads(path.read_text())
