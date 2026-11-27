"""Core data types shared by every pipeline stage.

Persisted records (documents, questions, chunks) are pydantic models so they are validated on
load. Per-query runtime objects are plain dataclasses for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QUERY_TYPES: tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "comparison",
    "temporal",
    "global",
    "creative",
    "null",
    "other",
)
ItemKind = Literal["chunk", "entity", "relation", "community"]


class Document(BaseModel):
    """One normalised corpus record (plan Phase 1, step 1)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = ""
    text: str
    source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document text is empty")
        return v


class Question(BaseModel):
    """One normalised evaluation question.

    `answers`, `gold_doc_ids` and `gold_evidence` are *offline-only* fields: the online pipeline
    receives a :class:`Query`, never a Question (FR-D5).
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    dataset: str
    question: str = Field(min_length=1)
    answers: list[str] = Field(default_factory=list)
    query_type: str = "other"
    answerable: bool = True
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_query(self) -> Query:
        return Query(query_id=self.question_id, text=self.question)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    index: int
    text: str
    n_tokens: int
    start_char: int
    end_char: int
    title: str = ""


@dataclass(frozen=True)
class Query:
    """What the online pipeline is allowed to see (no answers, no gold evidence)."""

    query_id: str
    text: str


@dataclass
class ContextItem:
    """A unit of retrieved context: a chunk, an entity profile, a relation or a community report."""

    item_id: str
    kind: ItemKind
    text: str
    score: float = 0.0
    source_chunk_ids: list[str] = field(default_factory=list)
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallRecord:
    """One model call (LLM, embedding batch or rerank batch) for cost & observability."""

    role: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None  # None = price unknown (reported as missing, never 0)
    cached: bool = False
    cost_is_estimate: bool = True
    tokens_are_estimate: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    mode: str
    items: list[ContextItem]
    timings: dict[str, float] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        """Distinct chunk IDs in rank order, including KG-item provenance."""
        seen: dict[str, None] = {}
        for it in self.items:
            ids = [it.item_id] if it.kind == "chunk" else it.source_chunk_ids
            for cid in ids:
                seen.setdefault(cid, None)
        return list(seen)

    @property
    def top_score(self) -> float | None:
        scores = [it.score for it in self.items if it.kind == "chunk"]
        return max(scores) if scores else None


@dataclass
class GeneratedAnswer:
    text: str
    refused: bool
    citations: list[int]
    cited_chunk_ids: list[str]
    invalid_citations: list[int]
    context_items: list[ContextItem]
    context_tokens: int
