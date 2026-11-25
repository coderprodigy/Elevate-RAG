"""Graph RAG pipelines: LightRAG retrieval modes, then the *same* generator as V1-V4.

| Pipeline   | LightRAG mode | Retrieves                                              |
|------------|---------------|--------------------------------------------------------|
| ``lr_naive``  | naive      | chunks by vector similarity (LightRAG's own V2)         |
| ``lr_local``  | local      | entities for low-level keywords + 1-hop neighbours      |
| ``lr_global`` | global     | relations for high-level keywords + their entities      |
| ``lr_hybrid`` | hybrid     | local + global                                          |
| ``lr_mix``    | mix        | graph + vector chunks (the "always fuse" baseline)      |

Shared with V1-V4 (protocol rule 1): generator prompt, chunk reranker (``rerank_chunks``),
``final_k`` for chunks, and the context token budget (LightRAG's own caps are set from it too).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hgrag.config import AppConfig, SearchSettings
from hgrag.data.schema import Query
from hgrag.generation.answers import clean_answer, is_refusal
from hgrag.generation.prompts import RAG_SYSTEM, RAG_VERSION, rag_prompt
from hgrag.graph.index import RETRIEVAL_SPEND, GraphIndex, SpendTally
from hgrag.graph.lightrag_adapter import KEYWORD_ROLE, GraphContext, unit_tokens
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import ContextUnit, Prediction
from hgrag.retrieval.budget import apply_token_budget
from hgrag.retrieval.rerank import Reranker

GENERATOR_ROLE = "generator"
_UNIT_TITLE = {"entity": "Knowledge graph: entity", "relation": "Knowledge graph: relation"}


class GraphPipeline:
    def __init__(
        self,
        mode: str,
        config: AppConfig,
        gateway: LLMGateway,
        graph: GraphIndex,
        *,
        reranker: Reranker | None,
        model_key: str | None = None,
        keyword_model_key: str | None = None,
        max_tokens: int = 512,
        use_cache: bool = True,
        settings: SearchSettings | None = None,
    ) -> None:
        self.name = f"lr_{mode}"
        self.mode = mode
        self._config = config
        self._gateway = gateway
        self._graph = graph
        self._reranker = reranker if config.graph.query.rerank_chunks else None
        self._model_key = model_key
        self._keyword_model_key = keyword_model_key
        self._max_tokens = max_tokens
        self._use_cache = use_cache
        self._settings = settings or config.retrieval.search

    # -- retrieval -------------------------------------------------------------------------------

    def query_params(self) -> dict[str, Any]:
        """LightRAG ``QueryParam`` fields; token caps derived from the shared budget."""
        q = self._config.graph.query
        budget = self._settings.context_token_budget
        return {
            "top_k": q.top_k,
            "chunk_top_k": q.chunk_top_k,
            "max_entity_tokens": int(budget * q.entity_token_share),
            "max_relation_tokens": int(budget * q.relation_token_share),
            "max_total_tokens": budget,
            "enable_rerank": False,  # we rerank chunks ourselves with the shared reranker
        }

    async def retrieve(self, question: str) -> list[ContextUnit]:
        context = await self._graph.retrieve(question, self.mode, self.query_params())
        return await asyncio.to_thread(self._select, question, context)

    def _select(self, question: str, context: GraphContext) -> list[ContextUnit]:
        chunks = context.chunks
        if self._reranker is not None and chunks:
            head = chunks[: self._settings.rerank_candidates]
            scores = self._reranker.score(question, [self._indexed_text(c) for c in head])
            chunks = [
                c.model_copy(update={"score": s})
                for c, s in sorted(zip(head, scores, strict=True), key=lambda p: -p[1])
            ]
        units = [*context.entities, *context.relations, *chunks[: self._settings.final_k]]
        return apply_token_budget(
            units, [unit_tokens(u) for u in units], self._settings.context_token_budget
        )

    def _indexed_text(self, unit: ContextUnit) -> str:
        chunk = self._graph.chunks.get(unit.unit_id)
        return chunk.indexed_text if chunk else unit.text

    def _title(self, unit: ContextUnit) -> str:
        if unit.kind in _UNIT_TITLE:
            return _UNIT_TITLE[unit.kind]
        chunk = self._graph.chunks.get(unit.unit_id)
        return chunk.title if chunk else ""

    # -- pipeline protocol -----------------------------------------------------------------------

    async def answer(self, query: Query) -> Prediction:
        tally = SpendTally()
        token = RETRIEVAL_SPEND.set(tally)
        started = time.perf_counter()
        try:
            units = await self.retrieve(query.question)
        finally:
            RETRIEVAL_SPEND.reset(token)
        retrieval_ms = (time.perf_counter() - started) * 1000

        response = await self._gateway.acomplete(
            GENERATOR_ROLE,
            self._prompt(query, units),
            system=RAG_SYSTEM,
            max_tokens=self._max_tokens,
            model_key=self._model_key,
            use_cache=self._use_cache,
        )
        answer = clean_answer(response.text)
        return Prediction(
            qid=query.qid,
            answer=answer,
            refused=is_refusal(answer),
            contexts=units,
            route=self.mode,
            model_key=response.model_key,
            input_tokens=response.input_tokens + response.cached_input_tokens + tally.input_tokens,
            output_tokens=response.output_tokens + tally.output_tokens,
            cost_usd=response.cost_usd + tally.cost_usd,
            latency_ms=retrieval_ms + response.latency_ms,
            cache_hit=response.cache_hit,
            status=response.status,
            meta={
                "retrieval_ms": round(retrieval_ms, 1),
                "retrieval_cost_usd": round(tally.cost_usd, 6),
                "n_entities": sum(u.kind == "entity" for u in units),
                "n_relations": sum(u.kind == "relation" for u in units),
                "n_chunks": sum(u.kind == "chunk" for u in units),
            },
        )

    def estimate_cost(self, query: Query) -> float:
        """Exact prompt if retrieval is cached; otherwise a full-budget prompt + keyword call."""
        cached = self._graph.cached(query.question, self.mode, self.query_params())
        if cached is not None:
            prompt = self._prompt(query, self._select(query.question, cached))
            keyword_cost = 0.0
        else:
            filler = "x" * (self._settings.context_token_budget * 3)  # ≈ budget tokens
            prompt = rag_prompt(query.question, [("", filler)])
            q = self._config.graph.query
            keyword_cost = (
                0.0
                if self.mode == "naive"
                else self._gateway.estimate_cost(
                    KEYWORD_ROLE,
                    "k" * (q.est_keyword_input_tokens * 3),
                    max_tokens=q.est_keyword_output_tokens,
                    model_key=self._keyword_model_key,
                    use_cache=False,
                )
            )
        return keyword_cost + self._gateway.estimate_cost(
            GENERATOR_ROLE,
            prompt,
            system=RAG_SYSTEM,
            max_tokens=self._max_tokens,
            model_key=self._model_key,
            use_cache=self._use_cache,
        )

    async def aclose(self) -> None:
        await self._graph.aclose()

    def describe(self) -> dict[str, Any]:
        return {
            "pipeline": self.name,
            "mode": self.mode,
            "generator_model": self._gateway.model_key_for(GENERATOR_ROLE, self._model_key),
            "keyword_model": self._gateway.model_key_for(KEYWORD_ROLE, self._keyword_model_key),
            "prompt": RAG_VERSION,
            "max_tokens": self._max_tokens,
            "reranker": self._reranker.name if self._reranker else None,
            "search": self._settings.model_dump(),
            "query_params": self.query_params(),
            **self._graph.describe(),
        }

    def _prompt(self, query: Query, units: list[ContextUnit]) -> str:
        return rag_prompt(query.question, [(self._title(u), u.text) for u in units])
