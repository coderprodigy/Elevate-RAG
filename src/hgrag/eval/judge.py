"""LLM-judge correctness (the primary answer metric): correct / partial / wrong vs the reference.

The judge model must come from a family that is not used by any generator (protocol rule 3);
it is the ``judge`` role in configs/models.yaml. Results go to ``judgments.jsonl`` in the run dir.
Refused answers to answerable questions are scored "wrong" without a call (saves money).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from hgrag.data.schema import Query
from hgrag.eval.batch import BatchOutcome, read_models, run_resumable
from hgrag.eval.llm_json import JSONExtractionError, extract_json
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import Prediction

JUDGE_ROLE = "judge"
JUDGE_PROMPT_VERSION = "judge/v1"
Verdict = Literal["correct", "partial", "wrong"]
VERDICT_SCORE: dict[Verdict, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM = """You grade an answer to a question against reference answer(s).
- correct: the answer states the same fact(s) as a reference (wording, extra true detail and \
citation markers like [1] do not matter). For long references, it covers the key points.
- partial: the answer is partly right (e.g. incomplete, too vague, or one of several required \
items is missing or wrong).
- wrong: the answer contradicts the references, answers a different question, or says it \
cannot answer.
Reply with JSON only: {"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}"""

_VERDICT_WORD = re.compile(r"\b(correct|partial|wrong)\b", re.IGNORECASE)


class Judgment(BaseModel):
    qid: str
    verdict: Verdict
    score: float
    reason: str
    judge_model: str  # "rule" when no call was made


def judge_prompt(query: Query, answer: str) -> str:
    references = "\n".join(f"- {ref}" for ref in query.answers)
    return f"Question: {query.question}\nReference answer(s):\n{references}\nAnswer: {answer}"


def parse_verdict(reply: str) -> tuple[Verdict, str]:
    """Read the verdict from JSON, falling back to the first verdict word in the reply."""
    try:
        data = extract_json(reply)
        verdict = str(data.get("verdict", "")).lower() if isinstance(data, dict) else ""
        reason = str(data.get("reason", "")) if isinstance(data, dict) else ""
    except JSONExtractionError:
        verdict, reason = "", reply.strip()[:200]
    if verdict not in VERDICT_SCORE:
        match = _VERDICT_WORD.search(reply)
        if match is None:
            raise ValueError(f"judge reply has no verdict: {reply[:120]!r}")
        verdict = match.group(1).lower()
    return verdict, reason  # type: ignore[return-value]


class AnswerJudge:
    def __init__(self, gateway: LLMGateway, *, model_key: str | None = None) -> None:
        self._gateway = gateway
        self._model_key = model_key

    def needs_call(self, prediction: Prediction) -> bool:
        return not prediction.refused

    async def judge(self, query: Query, prediction: Prediction) -> Judgment:
        if prediction.refused:
            return Judgment(
                qid=query.qid,
                verdict="wrong",
                score=0.0,
                reason="refused an answerable question",
                judge_model="rule",
            )
        response = await self._gateway.acomplete(
            JUDGE_ROLE,
            judge_prompt(query, prediction.answer),
            system=JUDGE_SYSTEM,
            max_tokens=200,
            model_key=self._model_key,
        )
        verdict, reason = parse_verdict(response.text)
        return Judgment(
            qid=query.qid,
            verdict=verdict,
            score=VERDICT_SCORE[verdict],
            reason=reason,
            judge_model=response.model_key,
        )

    def estimate_cost(self, query: Query, prediction: Prediction) -> float:
        if not self.needs_call(prediction):
            return 0.0
        return self._gateway.estimate_cost(
            JUDGE_ROLE,
            judge_prompt(query, prediction.answer),
            system=JUDGE_SYSTEM,
            max_tokens=200,
            model_key=self._model_key,
        )


def pending_judgments(
    queries: dict[str, Query], predictions: Sequence[Prediction], out_path: Path
) -> list[tuple[Query, Prediction]]:
    """Answerable questions with a prediction but no judgment yet."""
    done = {j.qid for j in read_models(out_path, Judgment)}
    return [
        (queries[p.qid], p)
        for p in predictions
        if p.qid in queries and queries[p.qid].answerable and p.qid not in done
    ]


async def judge_predictions(
    judge: AnswerJudge,
    pending: Sequence[tuple[Query, Prediction]],
    out_path: Path,
    *,
    concurrency: int = 8,
) -> BatchOutcome:
    async def work(pair: tuple[Query, Prediction]) -> Judgment:
        return await judge.judge(*pair)

    return await run_resumable(
        pending, work, out_path, concurrency=concurrency, label=lambda pair: pair[0].qid
    )
