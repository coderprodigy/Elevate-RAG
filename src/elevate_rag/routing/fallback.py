"""Routed answering, including R4 confidence fallback (FR-R4).

R4: run the cheap path; escalate **once** to `fallback_escalate_mode` when the top rerank score
is below the dev-tuned threshold or the answer is a refusal. The initial route, confidence,
reason, final route and the extra cost are all recorded on the QueryRecord.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from elevate_rag.config import RouterConfig
from elevate_rag.pipeline import ElevateRAG, QueryRecord
from elevate_rag.providers.base import collect_calls
from elevate_rag.routing.router import Router
from elevate_rag.types import Query


def should_escalate(rec: QueryRecord, cfg: RouterConfig) -> str | None:
    if rec.error:
        return "initial path failed"
    if cfg.fallback_on_refusal and rec.refused:
        return "answer was 'insufficient information'"
    if rec.top_score is None:
        return "no chunk retrieved"
    if rec.top_score < cfg.fallback_threshold:
        return f"top rerank score {rec.top_score:.3f} < threshold {cfg.fallback_threshold:.3f}"
    return None


async def answer_routed(system: ElevateRAG, query: Query, router: Router) -> QueryRecord:
    cfg = router.cfg
    if cfg.level == "fallback":
        first = await system.answer(query, cfg.fallback_cheap_mode)
        reason = should_escalate(first, cfg)
        route: dict[str, Any] = {
            "level": "R4",
            "initial_mode": first.mode,
            "confidence": first.top_score,
            "escalated": reason is not None,
            "reason": reason,
            "final_mode": first.mode,
        }
        if reason is None:
            first.route = route
            return first
        second = await system.answer(query, cfg.fallback_escalate_mode)
        route.update(
            final_mode=second.mode,
            initial_answer=first.answer,
            extra_prompt_tokens=second.prompt_tokens,
            extra_cost_usd=second.cost_usd,
        )
        second.route = route
        second.add_calls(first.calls)
        for k, v in first.timings.items():
            second.timings[f"initial_{k}"] = v
        return second

    with collect_calls() as route_calls:
        decision = await router.decide(query.text)
    rec = await system.answer(query, decision.mode)
    rec.route = decision.as_dict()
    if route_calls:
        from dataclasses import asdict

        rec.add_calls([asdict(c) for c in route_calls])
    return rec


def tune_threshold(
    top_scores: Sequence[float | None],
    cheap_correct: Sequence[bool],
    escalated_correct: Sequence[bool],
    refused: Sequence[bool] | None = None,
    *,
    on_refusal: bool = True,
) -> dict[str, float]:
    """Pick the threshold maximising dev accuracy (ties → fewer escalations).

    For each candidate threshold t, question i escalates iff score_i < t (or it refused); its
    outcome is then escalated_correct[i], else cheap_correct[i].
    """
    n = len(top_scores)
    if not (n == len(cheap_correct) == len(escalated_correct)):
        raise ValueError("inputs must be question-aligned")
    ref = list(refused) if refused is not None else [False] * n
    finite = sorted({s for s in top_scores if s is not None})
    candidates = [float("-inf"), *finite, float("inf")]
    best: tuple[float, float, float] | None = None
    for t in candidates:
        esc = [
            (s is None) or s < t or (on_refusal and r) for s, r in zip(top_scores, ref, strict=True)
        ]
        acc = (
            sum(
                (e2 if e else c)
                for e, c, e2 in zip(esc, cheap_correct, escalated_correct, strict=True)
            )
            / n
        )
        rate = sum(esc) / n
        if best is None or acc > best[1] or (acc == best[1] and rate < best[2]):
            best = (t, acc, rate)
    assert best is not None
    t = best[0]
    if t == float("-inf"):
        t = -1e9
    elif t == float("inf"):
        t = 1e9
    return {"threshold": t, "dev_accuracy": best[1], "escalation_rate": best[2]}
