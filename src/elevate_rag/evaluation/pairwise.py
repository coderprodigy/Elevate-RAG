"""Pairwise comparison of two finished runs on global / summary-style questions (FR-E7).

Each question is judged twice with the answer order swapped (PairwiseJudge); a side wins a
criterion only if both orders agree, otherwise it is a tie. Win rates are reported per criterion
with ties and invalid judgements counted explicitly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from elevate_rag.evaluation.judge import CRITERIA, PairwiseJudge
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.types import Question

GLOBAL_TYPES = ("global", "creative")


async def compare_runs(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    questions: Sequence[Question],
    judge_llm: BaseLLM,
    *,
    query_types: Sequence[str] = GLOBAL_TYPES,
    concurrency: int = 4,
) -> dict[str, Any]:
    qmap = {q.question_id: q for q in questions if q.query_type in query_types}
    a = {r["question_id"]: r for r in rows_a if r["question_id"] in qmap}
    b = {r["question_id"]: r for r in rows_b if r["question_id"] in qmap}
    qids = sorted(set(a) & set(b))
    judge = PairwiseJudge(judge_llm)
    sem = asyncio.Semaphore(concurrency)

    async def one(qid: str) -> dict[str, str]:
        async with sem:
            return await judge.compare(qmap[qid].question, a[qid]["answer"], b[qid]["answer"])

    results = await asyncio.gather(*(one(q) for q in qids))
    out: dict[str, Any] = {"n": len(qids), "query_types": list(query_types)}
    for c in CRITERIA:
        votes = [r[c] for r in results]
        n = len(votes) or 1
        out[c] = {
            "a_wins": votes.count("1") / n,
            "b_wins": votes.count("2") / n,
            "ties": votes.count("tie") / n,
            "invalid": votes.count("invalid"),
        }
    out["per_question"] = dict(zip(qids, results, strict=True))
    return out
