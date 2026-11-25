"""Error analysis: sample failures per system, suggest a cause, let a person confirm it.

Causes (plan "Error analysis"): retrieval miss, extraction miss, wrong route, context overload,
generation error, judge error. The suggestion is a heuristic from what the run recorded; the
human label is what counts in the report's failure-cause chart.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel

from hgrag.analysis.collect import RunRecord
from hgrag.eval.scoring import QuestionScore

CAUSES = (
    "retrieval_miss",
    "extraction_miss",
    "wrong_route",
    "context_overload",
    "generation_error",
    "judge_error",
)
_OVERLOAD_TOKEN_SHARE = 0.75


class FailureItem(BaseModel):
    run: str
    qid: str
    query_type: str
    suggested: str


class FailureTag(BaseModel):
    run: str
    qid: str
    labeller: str
    cause: str


def suggest_cause(
    score: QuestionScore, *, is_graph: bool, routed_away: bool, budget_tokens: int
) -> str:
    """Heuristic first guess, in the order a reviewer would check."""
    if routed_away:
        return "wrong_route"
    retrieval = score.chunk_retrieval or score.doc_retrieval
    found = None if retrieval is None else max((v or 0.0) for v in retrieval.recall.values())
    if found == 0.0:
        return "extraction_miss" if is_graph else "retrieval_miss"
    if found and score.tokens > _OVERLOAD_TOKEN_SHARE * budget_tokens:
        return "context_overload"
    return "generation_error"


def sample_failures(run: RunRecord, n: int, seed: int, budget_tokens: int) -> list[FailureItem]:
    failures = [s for s in run.scores if s.correct < 1.0]
    picked = random.Random(f"{seed}:{run.run_name}").sample(failures, min(n, len(failures)))
    is_graph = run.pipeline.startswith("lr_")
    return [
        FailureItem(
            run=run.run_name,
            qid=s.qid,
            query_type=s.query_type,
            suggested=suggest_cause(
                s, is_graph=is_graph, routed_away=False, budget_tokens=budget_tokens
            ),
        )
        for s in picked
    ]


def cause_counts(tags: Sequence[FailureTag]) -> dict[str, dict[str, int]]:
    """system (run) → cause → count, for the failure-cause chart."""
    table: dict[str, Counter[str]] = {}
    for tag in tags:
        table.setdefault(tag.run, Counter())[tag.cause] += 1
    return {run: {c: counts.get(c, 0) for c in CAUSES} for run, counts in table.items()}
