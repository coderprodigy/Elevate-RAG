"""V1-V4 vector RAG baselines (plan Phase 1 / E1):

| Variant | Retrieval     | Reranker |
|---------|---------------|----------|
| v1      | BM25          | no       |
| v2      | dense         | no       |
| v3      | hybrid (RRF)  | no       |
| v4      | hybrid (RRF)  | yes      |  ← reference vector baseline

All variants share the chunker, ``final_k``, the token budget and the generator prompt; only
the retrieval step differs (protocol rule 1).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hgrag.data.schema import Query
from hgrag.generation.answers import clean_answer, is_refusal
from hgrag.generation.prompts import RAG_SYSTEM, RAG_VERSION, rag_prompt
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import ContextUnit, Prediction
from hgrag.retrieval.searcher import Hit, RetrievalSpec, Searcher

GENERATOR_ROLE = "generator"

VARIANTS: dict[str, RetrievalSpec] = {
    "v1": RetrievalSpec(retriever="bm25", rerank=False),
    "v2": RetrievalSpec(retriever="dense", rerank=False),
    "v3": RetrievalSpec(retriever="hybrid", rerank=False),
    "v4": RetrievalSpec(retriever="hybrid", rerank=True),
}


class VectorPipeline:
    def __init__(
        self,
        name: str,
        gateway: LLMGateway,
        searcher: Searcher,
        *,
        model_key: str | None = None,
        max_tokens: int = 512,
        use_cache: bool = True,
    ) -> None:
        self._use_cache = use_cache
        self.name = name
        self._gateway = gateway
        self._searcher = searcher
        self._model_key = model_key
        self._max_tokens = max_tokens

    async def answer(self, query: Query) -> Prediction:
        started = time.perf_counter()
        hits = await asyncio.to_thread(self._searcher.search, query.question)
        retrieval_ms = (time.perf_counter() - started) * 1000

        response = await self._gateway.acomplete(
            GENERATOR_ROLE,
            self._prompt(query, hits),
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
            contexts=[
                ContextUnit(
                    unit_id=h.chunk.chunk_id,
                    kind="chunk",
                    text=h.chunk.text,
                    score=h.score,
                    source_doc_ids=[h.chunk.doc_id],
                )
                for h in hits
            ],
            model_key=response.model_key,
            input_tokens=response.input_tokens + response.cached_input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=retrieval_ms + response.latency_ms,
            cache_hit=response.cache_hit,
            status=response.status,
            meta={"retrieval_ms": round(retrieval_ms, 1)},
        )

    def estimate_cost(self, query: Query) -> float:
        """Exact prompt if retrieval is cached; otherwise a full-budget prompt (no retrieval,
        so dry runs stay instant even when reranking is slow)."""
        hits = self._searcher.cached(query.question)
        prompt = (
            self._prompt(query, hits)
            if hits is not None
            else rag_prompt(
                query.question, [("", "x" * (self._searcher.settings.context_token_budget * 3))]
            )
        )
        return self._gateway.estimate_cost(
            GENERATOR_ROLE,
            prompt,
            system=RAG_SYSTEM,
            max_tokens=self._max_tokens,
            model_key=self._model_key,
            use_cache=self._use_cache,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "pipeline": self.name,
            "generator_model": self._gateway.model_key_for(GENERATOR_ROLE, self._model_key),
            "prompt": RAG_VERSION,
            "max_tokens": self._max_tokens,
            **self._searcher.describe(),
        }

    @staticmethod
    def _prompt(query: Query, hits: list[Hit]) -> str:
        return rag_prompt(query.question, [(h.chunk.title, h.chunk.text) for h in hits])
