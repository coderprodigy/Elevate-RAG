"""Pairwise LLM judging for summary-style ("global") questions, with position-bias control.

Every comparison runs twice with the answer order swapped (protocol rule 4). A system wins only
if both orders agree; otherwise the result is a tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hgrag.eval.llm_json import JSONExtractionError, extract_json
from hgrag.llm.gateway import LLMGateway

PAIRWISE_PROMPT_VERSION = "pairwise/v1"
Criterion = Literal["comprehensiveness", "diversity", "empowerment"]
Winner = Literal["a", "b", "tie"]

CRITERIA: dict[Criterion, str] = {
    "comprehensiveness": "How much detail does the answer give to cover every aspect of the "
    "question?",
    "diversity": "How varied and rich are the perspectives and insights the answer provides?",
    "empowerment": "How well does the answer help the reader understand the topic and make "
    "informed judgements?",
}

PAIRWISE_SYSTEM = """You compare two answers to the same question on one criterion.
Judge only that criterion. Ignore answer length unless the criterion is about coverage, and \
ignore the order in which the answers are shown.
Reply with JSON only: {"winner": "1" | "2" | "tie", "reason": "<one sentence>"}"""


@dataclass(frozen=True)
class PairwiseResult:
    winner: Winner
    first_order: Winner  # verdict with (a, b) shown as (1, 2)
    swapped_order: Winner  # verdict with (b, a) shown as (1, 2), mapped back to a/b


def _prompt(question: str, criterion: Criterion, first: str, second: str) -> str:
    return (
        f"Criterion: {criterion}: {CRITERIA[criterion]}\n\nQuestion: {question}\n\n"
        f"Answer 1:\n{first}\n\nAnswer 2:\n{second}"
    )


def _parse(reply: str) -> Literal["1", "2", "tie"]:
    try:
        data = extract_json(reply)
        value = str(data.get("winner", "")).strip().lower() if isinstance(data, dict) else ""
    except JSONExtractionError:
        value = ""
    return value if value in ("1", "2") else "tie"  # type: ignore[return-value]


async def compare(
    gateway: LLMGateway,
    question: str,
    answer_a: str,
    answer_b: str,
    criterion: Criterion,
    *,
    model_key: str | None = None,
) -> PairwiseResult:
    async def ask(first: str, second: str) -> Literal["1", "2", "tie"]:
        response = await gateway.acomplete(
            "judge",
            _prompt(question, criterion, first, second),
            system=PAIRWISE_SYSTEM,
            max_tokens=150,
            model_key=model_key,
        )
        return _parse(response.text)

    first = {"1": "a", "2": "b", "tie": "tie"}[await ask(answer_a, answer_b)]
    swapped = {"1": "b", "2": "a", "tie": "tie"}[await ask(answer_b, answer_a)]
    winner = first if first == swapped else "tie"
    return PairwiseResult(winner, first, swapped)  # type: ignore[arg-type]


def win_rates(results: list[PairwiseResult]) -> dict[Winner, float]:
    """Share of comparisons won by a, won by b, and tied."""
    n = len(results) or 1
    winners: tuple[Winner, ...] = ("a", "b", "tie")
    return {w: sum(r.winner == w for r in results) / n for w in winners}
