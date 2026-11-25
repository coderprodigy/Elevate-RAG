"""The interface every QA pipeline implements (V0-V4, graph modes, routed pipelines).

The runner only knows this interface, so swapping a pipeline never touches evaluation code.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hgrag.data.schema import Query

UnitKind = Literal["chunk", "entity", "relation"]


class ContextUnit(BaseModel):
    """One piece of retrieved context shown to the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str
    kind: UnitKind
    text: str
    score: float | None = None
    source_doc_ids: list[str] = Field(default_factory=list)  # for doc-level retrieval metrics


class Prediction(BaseModel):
    """A pipeline's output for one query, as stored in ``predictions.jsonl``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    answer: str
    refused: bool  # the answer is the "insufficient information" refusal
    contexts: list[ContextUnit] = Field(default_factory=list)  # in rank order
    route: str | None = None  # retrieval path taken (router experiments)
    model_key: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    status: str  # "ok" | "refusal" (provider safety refusal) | "error"
    meta: dict[str, Any] = Field(default_factory=dict)


class Pipeline(Protocol):
    name: str

    async def answer(self, query: Query) -> Prediction:
        """Answer one query (retrieval, if any, plus generation)."""
        ...

    def estimate_cost(self, query: Query) -> float:
        """Upper-bound USD to answer ``query`` now (0 if fully cached). Used by ``--dry-run``."""
        ...

    def describe(self) -> dict[str, Any]:
        """What defines this pipeline (models, prompts, k...); written to the run config."""
        ...
