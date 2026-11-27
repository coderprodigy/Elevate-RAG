"""The Elevate-RAG system: one shared index, every retrieval mode, one generation path.

Modes
-----
* Vector baselines (plan Phase 1): ``v0`` closed-book, ``v1`` BM25, ``v2`` dense, ``v3`` hybrid
  RRF, ``v4`` hybrid RRF + cross-encoder (alias ``vector``).
* LightRAG modes (plan Phase 2): ``naive``, ``local``, ``global``, ``hybrid``, ``mix``.

Every mode except v0–v3 passes through the shared reranker; every mode passes through the
shared token budget and the same generation prompt (plan protocol rule 1, D-07).

Index layout: ``<index_dir>/<dataset>/<index_key>/`` with ``index_manifest.json``,
``chunks.jsonl``, ``dense/``, ``lightrag/``, ``chunk_map.json``, ``resolution.json``,
``graph_quality.json`` and optionally ``community/``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from elevate_rag import __version__
from elevate_rag.cache import KVCache
from elevate_rag.config import PipelineConfig
from elevate_rag.generation.context import assemble_context
from elevate_rag.generation.generator import PROMPT_VERSION, Generator
from elevate_rag.graph.community import CommunityIndex
from elevate_rag.graph.lightrag_engine import GRAPH_MODES, LightRAGEngine
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.indexing.chunker import Chunker
from elevate_rag.providers.base import UsageLedger, collect_calls
from elevate_rag.providers.embeddings import BaseEmbedder, make_embedder
from elevate_rag.providers.llm import BaseLLM, make_llm
from elevate_rag.providers.rerank import BaseReranker, make_reranker
from elevate_rag.retrieval.lexical import BM25Index, reciprocal_rank_fusion
from elevate_rag.storage.vector import ExactStore, VectorStore, make_vector_store
from elevate_rag.types import CallRecord, ContextItem, Document, Query, RetrievalResult
from elevate_rag.utils import (
    BudgetExceededError,
    ConfigError,
    ElevateError,
    Tokenizer,
    write_json,
)

log = logging.getLogger(__name__)

VECTOR_MODES = ("v0", "v1", "v2", "v3", "v4")
ALL_MODES = VECTOR_MODES + GRAPH_MODES
MODE_ALIASES = {"vector": "v4", "closed_book": "v0", "bm25": "v1", "dense": "v2"}
# Static relative cost order (D-08); measured tokens override it in the oracle.
MODE_COST_ORDER = {
    "v0": 0,
    "v1": 1,
    "v2": 1,
    "v3": 2,
    "v4": 3,
    "naive": 3,
    "local": 4,
    "global": 4,
    "hybrid": 5,
    "mix": 6,
}


def canonical_mode(mode: str) -> str:
    m = MODE_ALIASES.get(mode, mode)
    if m not in ALL_MODES:
        raise ConfigError(f"unknown mode {mode!r}; choose from {ALL_MODES + tuple(MODE_ALIASES)}")
    return m


@dataclass
class QueryRecord:
    """Everything recorded for one question under one configuration (FR-O2)."""

    question_id: str
    question: str
    mode: str
    answer: str = ""
    refused: bool = False
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)  # ranked, after rerank
    candidate_chunk_ids: list[str] = field(default_factory=list)  # pre-rerank candidates
    context_item_ids: list[str] = field(default_factory=list)  # exactly what the LLM saw
    context_items: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    top_score: float | None = None
    context_tokens: int = 0
    route: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0
    error: str | None = None

    def finalize(self, calls: list[CallRecord]) -> QueryRecord:
        self.calls = [asdict(c) for c in calls]
        return self._recompute()

    def add_calls(self, calls: list[dict[str, Any]]) -> QueryRecord:
        """Merge extra calls (router LLM, R4 initial path) into this record's totals."""
        self.calls = list(calls) + self.calls
        return self._recompute()

    def _recompute(self) -> QueryRecord:
        llm = [c for c in self.calls if not c["role"].startswith(("embed", "rerank"))]
        # Token totals count every LLM call made for this query, cached or not: they describe
        # the *workload*; cost_usd only counts uncached calls (actual spend).
        self.prompt_tokens = sum(c["prompt_tokens"] for c in llm)
        self.completion_tokens = sum(c["completion_tokens"] for c in llm)
        costs = [c["cost_usd"] for c in self.calls if not c["cached"]]
        self.cost_usd = None if any(c is None for c in costs) else float(sum(costs))
        return self


class ElevateRAG:
    def __init__(self, cfg: PipelineConfig, *, ledger: UsageLedger | None = None) -> None:
        if cfg.dataset is None:
            raise ConfigError("pipeline config needs `dataset`")
        self.cfg = cfg
        self.ledger = ledger or UsageLedger(max_usd=cfg.budget.max_usd)
        cache_dir = Path(cfg.cache_dir)
        m = cfg.models
        self.embedder: BaseEmbedder = make_embedder(
            m.embedder, ledger=self.ledger, cache=KVCache(cache_dir, "embeddings")
        )
        self.reranker: BaseReranker | None = make_reranker(
            m.reranker, ledger=self.ledger, cache=KVCache(cache_dir, "rerank")
        )
        llm_cache = KVCache(cache_dir, "llm")
        self.generator_llm: BaseLLM = make_llm(
            m.generator, role="generator", ledger=self.ledger, cache=llm_cache
        )
        self.extractor: BaseLLM | None = (
            make_llm(m.extractor, role="extractor", ledger=self.ledger, cache=llm_cache)
            if m.extractor
            else None
        )
        self.keyword_llm: BaseLLM | None = (
            make_llm(m.keyword, role="keyword", ledger=self.ledger, cache=llm_cache)
            if m.keyword
            else None
        )
        self.generator = Generator(self.generator_llm)
        self.tok = Tokenizer(cfg.chunking.encoding)
        self.index_path = Path(cfg.index_dir) / cfg.dataset / cfg.index_key()
        self.store: ChunkStore | None = None
        self.bm25: BM25Index | None = None
        self.dense: VectorStore | None = None
        self.graph: LightRAGEngine | None = None
        self.communities: CommunityIndex | None = None

    # =============================================================================== indexing
    def _dense_store(self, dim: int) -> VectorStore:
        name = f"{self.cfg.dataset}_{self.cfg.index_key()}_chunks"
        return make_vector_store(self.cfg.vector_store, name, dim)

    async def build_index(
        self, docs: list[Document], *, graph: bool = True, dataset_version: str | None = None
    ) -> dict[str, Any]:
        """Idempotent and resumable: chunking and dense vectors are cheap and deterministic;
        graph extraction resumes per document (LightRAG progress file)."""
        if not docs:
            raise ElevateError("cannot index an empty corpus")
        p = self.index_path
        p.mkdir(parents=True, exist_ok=True)
        t: dict[str, float] = {}
        before = self.ledger.summary()

        t0 = time.perf_counter()
        store = ChunkStore(Chunker(self.cfg.chunking).chunk_documents(docs))
        store.save(p / "chunks.jsonl")
        t["chunking_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ids = store.ids
        vecs = self.embedder.embed([store[i].text for i in ids], kind="document")
        dense = self._dense_store(vecs.shape[1])
        if dense.count() != len(ids):
            dense.add(ids, vecs, [{"doc_id": store[i].doc_id, "source": "chunk"} for i in ids])
        dense.save(p / "dense")
        t["dense_index_s"] = time.perf_counter() - t0
        self.store, self.dense = store, dense
        self.bm25 = BM25Index(self.cfg.retrieval.bm25_k1, self.cfg.retrieval.bm25_b).build(
            ids, [store[i].text for i in ids]
        )

        graph_info: dict[str, Any] = {"built": False}
        if graph:
            t0 = time.perf_counter()
            engine = await self._open_graph()
            graph_info = await engine.index(docs, store, p / "lightrag_progress.txt")
            engine.save_chunk_map(p / "chunk_map.json")
            if self.cfg.resolution.enabled:
                graph_info["resolution"] = await self._resolve_entities(engine)
            if self.cfg.community.enabled:
                graph_info["communities"] = await self._build_communities(engine)
            graph_info["built"] = not graph_info.get("failed")
            t["graph_index_s"] = time.perf_counter() - t0

        after = self.ledger.summary()
        manifest = {
            "elevate_rag_version": __version__,
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "dataset": self.cfg.dataset,
            "dataset_version": dataset_version,
            "index_key": self.cfg.index_key(),
            "config_hash": self.cfg.config_hash(),
            "config": self.cfg.model_dump(mode="json"),
            "embedder": {"id": self.embedder.model_id, "dimension": int(vecs.shape[1])},
            "vector_store": {
                "backend": dense.backend,
                "hnsw": {
                    "M": self.cfg.vector_store.hnsw_m,
                    "ef_construction": self.cfg.vector_store.hnsw_ef_construction,
                    "ef_search": self.cfg.vector_store.hnsw_ef_search,
                },
            },
            "documents": len(docs),
            "chunks": len(store),
            "chunk_tokens_total": sum(c.n_tokens for c in store.chunks()),
            "graph": graph_info,
            "timings": t,
            "usage_delta": _usage_delta(before, after),
            "storage_bytes": _dir_size(p),
            "demo": self.cfg.demo,
        }
        write_json(p / "index_manifest.json", manifest)
        return manifest

    async def _resolve_entities(self, engine: LightRAGEngine) -> dict[str, Any]:
        from elevate_rag.graph.resolution import plan_merges

        nodes, _ = await engine.export_graph()
        plan = plan_merges(
            nodes,
            self.embedder,
            self.cfg.resolution.similarity_threshold,
            max_merges=self.cfg.resolution.max_merges,
        )
        applied, failed = 0, []
        for g in plan.groups:
            try:
                await engine.merge_entities(g["sources"], g["target"])
                applied += len(g["sources"])
            except Exception as exc:  # recorded, not swallowed: reported in the manifest
                failed.append({"group": g, "error": str(exc)[:300]})
        write_json(self.index_path / "resolution.json", {"groups": plan.groups, "failed": failed})
        return {"groups": len(plan.groups), "merged_entities": applied, "failed": len(failed)}

    async def _build_communities(self, engine: LightRAGEngine) -> dict[str, Any]:
        from elevate_rag.graph.community import build_reports, detect_communities

        if self.extractor is None:
            raise ConfigError("community reports need models.extractor")
        nodes, edges = await engine.export_graph()
        comms = detect_communities(nodes, edges, self.cfg.community)
        reports = await build_reports(comms, nodes, edges, self.extractor)
        idx = CommunityIndex.build(reports, self.embedder)
        idx.save(self.index_path / "community")
        self.communities = idx
        return {"communities": len(comms), "reports": len(reports)}

    # =============================================================================== loading
    async def _open_graph(self) -> LightRAGEngine:
        if self.graph is None:
            self.graph = await LightRAGEngine(
                self.cfg,
                self.index_path / "lightrag",
                embedder=self.embedder,
                extractor=self.extractor,
                keyword_llm=self.keyword_llm,
                query_llm=self.generator_llm,
            ).open()
        return self.graph

    async def open(self, *, graph: bool = True) -> ElevateRAG:
        p = self.index_path
        if not (p / "index_manifest.json").exists():
            raise ElevateError(f"no index at {p}; run `elevate-rag index build` first")
        manifest = json.loads((p / "index_manifest.json").read_text())
        if manifest["embedder"]["id"] != self.embedder.model_id:
            raise ElevateError("index was built with a different embedder (FR-M5)")
        self.store = ChunkStore.load(p / "chunks.jsonl")
        ids = self.store.ids
        self.bm25 = BM25Index(self.cfg.retrieval.bm25_k1, self.cfg.retrieval.bm25_b).build(
            ids, [self.store[i].text for i in ids]
        )
        dim = int(manifest["embedder"]["dimension"])
        if self.cfg.vector_store.backend == "exact":
            self.dense = ExactStore.load(
                self.cfg.vector_store, self._dense_store_name(), p / "dense"
            )
        else:
            self.dense = self._dense_store(dim)
            if self.dense.count() != len(ids):
                raise ElevateError(
                    f"{self.dense.backend} holds {self.dense.count()} vectors, "
                    f"index has {len(ids)} chunks — rebuild the index"
                )
        if graph and not manifest["graph"].get("built"):
            raise ElevateError(
                f"graph index requested but not built at {p} "
                "(run `elevate-rag index build` with --graph)"
            )
        if graph:
            engine = await self._open_graph()
            engine.load_chunk_map(p / "chunk_map.json")
        self.communities = CommunityIndex.load(p / "community")
        return self

    def _dense_store_name(self) -> str:
        return f"{self.cfg.dataset}_{self.cfg.index_key()}_chunks"

    async def close(self) -> None:
        if self.graph is not None:
            await self.graph.close()
            self.graph = None
        for llm in (self.generator_llm, self.extractor, self.keyword_llm):
            if llm is not None:
                await llm.aclose()
        if self.dense is not None:
            self.dense.close()

    # =============================================================================== retrieval
    def _chunk_item(self, cid: str, score: float) -> ContextItem:
        assert self.store is not None
        c = self.store[cid]
        return ContextItem(item_id=cid, kind="chunk", text=c.text, score=score, doc_id=c.doc_id)

    async def retrieve(self, query: Query, mode: str) -> RetrievalResult:
        mode = canonical_mode(mode)
        if self.store is None or self.bm25 is None or self.dense is None:
            raise ElevateError("pipeline not opened; call open() or build_index() first")
        rc = self.cfg.retrieval
        timings: dict[str, float] = {}
        if mode == "v0":
            return RetrievalResult(mode=mode, items=[])
        t0 = time.perf_counter()
        if mode in ("v1", "v2", "v3", "v4"):
            bm = self.bm25.search(query.text, rc.candidates) if mode != "v2" else []
            dn: list[tuple[str, float]] = []
            if mode != "v1":
                qv = self.embedder.embed([query.text], kind="query")[0]
                dn = self.dense.search(qv, rc.candidates)
            hits = {"v1": bm, "v2": dn}.get(mode) or reciprocal_rank_fusion(
                [bm, dn], k=rc.rrf_k, limit=rc.candidates
            )
            items = [self._chunk_item(cid, s) for cid, s in hits]
            res = RetrievalResult(mode=mode, items=items)
        else:
            if self.graph is None:
                raise ElevateError(f"mode {mode} needs the graph index")
            res = await self.graph.retrieve(query.text, mode, self.store)
            if (
                self.cfg.community.enabled
                and self.communities
                and mode in ("global", "hybrid", "mix")
            ):
                qv = self.embedder.embed([query.text], kind="query")[0]
                res.items = self.communities.search(qv, self.cfg.community.top_reports) + res.items
        timings["retrieval"] = time.perf_counter() - t0
        res.metadata["candidate_chunk_ids"] = [it.item_id for it in res.items if it.kind == "chunk"]
        if self.reranker is not None and mode not in ("v1", "v2", "v3"):
            t0 = time.perf_counter()
            res.items = self._rerank(query.text, res.items)
            timings["rerank"] = time.perf_counter() - t0
            res.metadata["reranked"] = True
        res.timings.update(timings)
        return res

    def _rerank(self, query: str, items: list[ContextItem]) -> list[ContextItem]:
        assert self.reranker is not None
        chunks = [it for it in items if it.kind == "chunk"]
        others = [it for it in items if it.kind != "chunk"]
        scores = self.reranker.score(query, [c.text for c in chunks])
        for c, s in zip(chunks, scores, strict=True):
            c.metadata["retrieval_score"] = c.score
            c.score = s
        chunks.sort(key=lambda c: (-c.score, c.item_id))
        return others + chunks

    # =============================================================================== answering
    async def answer(self, query: Query, mode: str) -> QueryRecord:
        mode = canonical_mode(mode)
        rec = QueryRecord(question_id=query.query_id, question=query.text, mode=mode)
        with collect_calls() as calls:
            try:
                res = await self.retrieve(query, mode)
                rec.timings.update(res.timings)
                rc = self.cfg.retrieval
                if mode == "v0":
                    items = None
                else:
                    ctx = assemble_context(
                        res.items,
                        budget=rc.context_budget,
                        final_k=rc.final_k,
                        kg_share=rc.kg_share,
                        tokenizer=self.tok,
                    )
                    items = ctx.items
                    rec.context_tokens = ctx.tokens
                    ranked = [it for it in res.items if it.kind == "chunk"]
                    rec.retrieved_chunk_ids = list(
                        dict.fromkeys([it.item_id for it in ranked] + res.chunk_ids)
                    )
                    rec.candidate_chunk_ids = res.metadata.get("candidate_chunk_ids", [])
                    rec.scores = [round(it.score, 6) for it in ranked[: rc.final_k]]
                    rec.top_score = res.top_score
                    rec.context_item_ids = [it.item_id for it in items]
                    rec.context_items = [
                        {
                            "id": it.item_id,
                            "kind": it.kind,
                            "source_chunk_ids": it.source_chunk_ids[:20],
                        }
                        for it in items
                    ]
                t0 = time.perf_counter()
                gen = await self.generator.answer(query.text, items, rec.context_tokens)
                rec.timings["generation"] = time.perf_counter() - t0
                rec.answer, rec.refused = gen.text, gen.refused
                rec.citations, rec.invalid_citations = gen.citations, gen.invalid_citations
                rec.cited_chunk_ids = gen.cited_chunk_ids
            except BudgetExceededError:
                raise  # a spending-cap breach aborts the run, it is not a per-question failure
            except ElevateError as exc:
                rec.error = f"{type(exc).__name__}: {exc}"
                log.error("query %s mode %s failed: %s", query.query_id, mode, exc)
        return rec.finalize(calls)


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for role, agg in after["by_role"].items():
        prev = before["by_role"].get(role, {})
        out[role] = {k: round(v - prev.get(k, 0), 6) for k, v in agg.items()}
    return {"spent_usd": round(after["spent_usd"] - before["spent_usd"], 6), "by_role": out}


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


__all__ = [
    "ALL_MODES",
    "MODE_COST_ORDER",
    "PROMPT_VERSION",
    "ElevateRAG",
    "QueryRecord",
    "canonical_mode",
]
