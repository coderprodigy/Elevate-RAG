"""V0: closed-book generation, no retrieval. Measures what the generator already knows."""

from __future__ import annotations

from typing import Any

from hgrag.data.schema import Query
from hgrag.generation.answers import clean_answer, is_refusal
from hgrag.generation.prompts import CLOSED_BOOK_SYSTEM, CLOSED_BOOK_VERSION, closed_book_prompt
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import Prediction

GENERATOR_ROLE = "generator"


class ClosedBookPipeline:
    name = "v0_closed_book"

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model_key: str | None = None,
        max_tokens: int = 256,
        use_cache: bool = True,
    ) -> None:
        self._use_cache = use_cache
        self._gateway = gateway
        self._model_key = model_key
        self._max_tokens = max_tokens

    async def answer(self, query: Query) -> Prediction:
        response = await self._gateway.acomplete(
            GENERATOR_ROLE,
            closed_book_prompt(query.question),
            system=CLOSED_BOOK_SYSTEM,
            max_tokens=self._max_tokens,
            model_key=self._model_key,
            use_cache=self._use_cache,
        )
        answer = clean_answer(response.text)
        return Prediction(
            qid=query.qid,
            answer=answer,
            refused=is_refusal(answer),
            model_key=response.model_key,
            input_tokens=response.input_tokens + response.cached_input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            cache_hit=response.cache_hit,
            status=response.status,
        )

    def estimate_cost(self, query: Query) -> float:
        return self._gateway.estimate_cost(
            GENERATOR_ROLE,
            closed_book_prompt(query.question),
            system=CLOSED_BOOK_SYSTEM,
            max_tokens=self._max_tokens,
            model_key=self._model_key,
            use_cache=self._use_cache,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "pipeline": self.name,
            "generator_model": self._gateway.model_key_for(GENERATOR_ROLE, self._model_key),
            "prompt": CLOSED_BOOK_VERSION,
            "max_tokens": self._max_tokens,
            "retrieval": None,
        }
