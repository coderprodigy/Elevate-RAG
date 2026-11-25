"""Normalised record types shared by every dataset.

Every loader converts its source format into these two types, so later phases never see
dataset-specific fields (those go into ``meta``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Cross-dataset query-type vocabulary; the router's query-type labels use it too.
QueryType = Literal[
    "single_hop",  # one fact from one passage
    "multi_hop",  # chain of facts across passages (bridge / inference / complex reasoning)
    "comparison",  # compare two entities
    "temporal",  # ordering or time arithmetic
    "global",  # corpus-wide summary / sensemaking
    "creative",  # GraphRAG-Bench level 4
    "unanswerable",  # the corpus does not contain the answer
]
QUERY_TYPES: tuple[QueryType, ...] = QueryType.__args__  # type: ignore[attr-defined]

Split = Literal["dev", "test"]


class Document(BaseModel):
    """One retrievable source unit (a passage, paragraph, article or book) before chunking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    text: str
    source: str  # dataset name
    meta: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """One question with its references and gold evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    question: str
    answers: list[str]  # acceptable short answers; empty for unanswerable questions
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)  # evidence sentences, when given
    query_type: QueryType
    native_type: str  # the dataset's own label, kept for per-dataset breakdowns
    hop_count: int | None = None
    answerable: bool = True
    split: Split | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
