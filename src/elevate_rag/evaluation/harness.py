"""Evaluation harness: run any pipeline/mode/router over any prepared dataset, then score.

Stage 1 `run_questions` — online; writes one JSONL row per question as soon as it finishes
(crash-safe, resumable: finished question IDs are skipped on restart; rows with errors are
retried only when `retry_failed=True`). Bounded concurrency.

Stage 2 `score_rows` — offline; adds metrics using the gold fields the online stage never saw.

Stage 3 `summarize` — aggregate means with bootstrap 95 % CIs, per-query-type breakdowns,
efficiency and error rates. Failed questions are counted (as wrong) and reported.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from elevate_rag.evaluation import llm_metrics
from elevate_rag.evaluation.judge import CorrectnessJudge
from elevate_rag.evaluation.metrics import (
    evidence_recall,
    exact_match,
    max_f1,
    mean_defined,
    ndcg_at_k,
    recall_at_k,
    short_answer,
)
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.pipeline import QueryRecord
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.stats import bootstrap_ci
from elevate_rag.types import Query, Question
from elevate_rag.utils import append_jsonl, read_jsonl, repair_jsonl_tail

log = logging.getLogger(__name__)

AnswerFn = Callable[[Query], Awaitable[QueryRecord]]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path, skip_corrupt_tail=True)) if path.exists() else []


async def run_questions(
    answer_fn: AnswerFn,
    questions: Sequence[Question],
    out_path: Path,
    *,
    run_meta: dict[str, Any],
    concurrency: int = 4,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Execute `answer_fn` for each question, appending rows to `out_path` (resumable)."""
    repair_jsonl_tail(out_path)  # a crash mid-write must not corrupt the rows appended next
    existing = {r["question_id"]: r for r in load_rows(out_path)}
    todo = [
        q
        for q in questions
        if q.question_id not in existing or (retry_failed and existing[q.question_id].get("error"))
    ]
    if len(todo) < len(questions):
        log.info(
            "resuming: %d/%d questions already done", len(questions) - len(todo), len(questions)
        )
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(q: Question) -> None:
        async with sem:
            rec = await answer_fn(q.to_query())  # the online stage sees only the Query
        row = {**run_meta, **asdict(rec)}
        async with lock:
            append_jsonl(out_path, row)

    await asyncio.gather(*(one(q) for q in todo))
    rows = {r["question_id"]: r for r in load_rows(out_path)}  # last write wins on retry
    return [rows[q.question_id] for q in questions if q.question_id in rows]


async def score_rows(
    rows: Sequence[dict[str, Any]],
    questions: Sequence[Question],
    store: ChunkStore | None,
    *,
    judge_llm: BaseLLM | None = None,
    metrics_llm: BaseLLM | None = None,
    ks: Sequence[int] = (5, 10, 50),
    concurrency: int = 4,
    string_evidence_valid: bool = True,
) -> list[dict[str, Any]]:
    """`string_evidence_valid=False` (paraphrased gold evidence) reports evidence recall as
    undefined instead of a misleadingly low number."""
    qmap = {q.question_id: q for q in questions}
    judge = CorrectnessJudge(judge_llm) if judge_llm else None
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        q = qmap[row["question_id"]]
        failed = bool(row.get("error"))
        ans = "" if failed else row["answer"]
        refused = bool(row.get("refused")) and not failed
        retrieved = row.get("retrieved_chunk_ids") or []
        s: dict[str, Any] = {
            "query_type": q.query_type,
            "answerable": q.answerable,
            "failed": failed,
        }
        s["em"] = exact_match(short_answer(ans), q.answers) if q.answerable else None
        s["f1"] = max_f1(short_answer(ans), q.answers) if q.answerable else None
        s["refusal_correct"] = refused if not q.answerable else not refused
        for k in ks:
            s[f"recall@{k}"] = recall_at_k(retrieved, q.gold_doc_ids, k)
        s["ndcg@10"] = ndcg_at_k(retrieved, q.gold_doc_ids, 10)
        context_ids = [i for i in row.get("context_item_ids", []) if store and i in store]
        texts = [store[i].text for i in context_ids] if store else []
        if not texts and row.get("context_texts"):  # external systems (predictions contract)
            texts = list(row["context_texts"])
        s["evidence_recall"] = (
            evidence_recall(texts, q.gold_evidence) if string_evidence_valid else None
        )
        async with sem:
            if judge is not None:
                v = await judge.judge(
                    q.question, q.answers, ans, answerable=q.answerable, refused=refused
                )
                s["judge"] = v.verdict
                s["judge_score"] = v.score
                s["correct"] = None if v.verdict == "invalid" else v.verdict == "correct"
            else:
                s["correct"] = bool(s["em"]) if q.answerable else s["refusal_correct"]
            if metrics_llm is not None and not failed:
                ref = q.answers[0] if q.answers else ""
                s["faithfulness"] = (
                    await llm_metrics.faithfulness(metrics_llm, q.question, ans, texts)
                    if not refused
                    else None
                )
                s["context_recall"] = await llm_metrics.context_recall(
                    metrics_llm, q.question, ref, texts
                )
                s["context_precision"] = await llm_metrics.context_precision(
                    metrics_llm, q.question, ref, texts
                )
        if failed:
            s["correct"] = False
        return {**row, "scores": s}

    return list(await asyncio.gather(*(one(r) for r in rows)))


def _ci(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ci = bootstrap_ci(values)
    return {"mean": ci.estimate, "ci_low": ci.low, "ci_high": ci.high, "n": ci.n}


def summarize(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"n": 0}
    keys = sorted(
        {
            k
            for r in scored
            for k, v in r["scores"].items()
            if isinstance(v, int | float | bool) or v is None
        }
        - {"answerable", "failed", "query_type"}
    )

    def block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rows)}
        for k in keys:
            vals = [r["scores"].get(k) for r in rows]
            defined = [float(v) for v in vals if v is not None]
            out[k] = _ci(defined)
            out[f"{k}__undefined"] = len(vals) - len(defined)
        return out

    overall = block(scored)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_type[r["scores"]["query_type"]].append(r)
    lat = [
        sum(v for k, v in r.get("timings", {}).items()) for r in scored if not r["scores"]["failed"]
    ]
    costs = [r.get("cost_usd") for r in scored]
    return {
        "overall": overall,
        "by_query_type": {t: block(rs) for t, rs in sorted(by_type.items())},
        "efficiency": {
            "latency_p50_s": float(np.percentile(lat, 50)) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens", 0) for r in scored])),
            "completion_tokens_mean": float(
                np.mean([r.get("completion_tokens", 0) for r in scored])
            ),
            "context_tokens_mean": float(np.mean([r.get("context_tokens", 0) for r in scored])),
            "cost_per_1k_queries_usd": (
                1000 * float(np.mean([c for c in costs if c is not None]))
                if costs and all(c is not None for c in costs)
                else None
            ),
            "cost_known": all(c is not None for c in costs),
        },
        "error_rate": sum(r["scores"]["failed"] for r in scored) / len(scored),
        "refusal_rate": sum(bool(r.get("refused")) for r in scored) / len(scored),
        "invalid_citation_rate": sum(bool(r.get("invalid_citations")) for r in scored)
        / len(scored),
        "judge_invalid": sum(r["scores"].get("judge") == "invalid" for r in scored),
    }


def per_question_correct(scored: Sequence[dict[str, Any]]) -> dict[str, bool]:
    """question_id → correct (invalid judge verdicts excluded) for paired tests and the oracle."""
    return {
        r["question_id"]: bool(r["scores"]["correct"])
        for r in scored
        if r["scores"].get("correct") is not None
    }


def write_scored(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = [
    "load_rows",
    "mean_defined",
    "per_question_correct",
    "run_questions",
    "score_rows",
    "summarize",
    "write_scored",
]
