"""LLM-checked context metrics, both built on one "is this statement supported?" step.

* **Faithfulness** (RAGAS definition): share of the *answer's* claims supported by the retrieved
  context. Step 1 splits the answer into atomic claims; step 2 checks each claim.
* **Evidence recall** (GraphRAG-Bench retrieval metric): share of the question's *gold evidence*
  sentences supported by the retrieved context. Needed because GraphRAG-Bench evidence is
  paraphrased, so it cannot be string-matched to chunks.

Implemented here (not with the ``ragas`` package) so every call is metered and cached.
Results are None when there is nothing to check (refusal, no context, or no evidence).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from hgrag.data.schema import Query
from hgrag.eval.batch import BatchOutcome, read_models, run_resumable
from hgrag.eval.llm_json import extract_json
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import Prediction

PROMPT_VERSION = "context_metrics/v1"
ROLE = "judge"
ContextMetric = Literal["faithfulness", "evidence_recall"]

CLAIMS_SYSTEM = """Split the answer into short, self-contained factual claims (resolve pronouns).
Ignore citation markers like [1]. Reply with JSON only: {"claims": ["...", "..."]}"""

VERIFY_SYSTEM = """For each numbered statement, decide whether the context directly supports it.
Use only the context, not your own knowledge.
Reply with JSON only: {"supported": [true, false, ...]} (one boolean per statement, in order)"""


class ContextScore(BaseModel):
    qid: str
    metric: ContextMetric
    score: float | None  # supported / checked; None if there was nothing to check
    n_checked: int
    n_supported: int


def _context_text(prediction: Prediction) -> str:
    return "\n\n".join(f"[{i}] {unit.text}" for i, unit in enumerate(prediction.contexts, 1))


async def _count_supported(
    gateway: LLMGateway, prediction: Prediction, statements: list[str], model_key: str | None
) -> int:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    reply = await gateway.acomplete(
        ROLE,
        f"Context:\n{_context_text(prediction)}\n\nStatements:\n{numbered}",
        system=VERIFY_SYSTEM,
        max_tokens=300,
        model_key=model_key,
    )
    verdicts = [bool(v) for v in extract_json(reply.text).get("supported", [])]
    return sum(verdicts[: len(statements)])  # missing verdicts count as unsupported


def _empty(prediction: Prediction, metric: ContextMetric) -> ContextScore:
    return ContextScore(qid=prediction.qid, metric=metric, score=None, n_checked=0, n_supported=0)


async def score_faithfulness(
    gateway: LLMGateway, question: str, prediction: Prediction, *, model_key: str | None = None
) -> ContextScore:
    if prediction.refused or not prediction.contexts:
        return _empty(prediction, "faithfulness")
    reply = await gateway.acomplete(
        ROLE,
        f"Question: {question}\nAnswer: {prediction.answer}",
        system=CLAIMS_SYSTEM,
        max_tokens=600,
        model_key=model_key,
    )
    claims = [str(c) for c in extract_json(reply.text).get("claims", []) if str(c).strip()]
    if not claims:
        return _empty(prediction, "faithfulness")
    supported = await _count_supported(gateway, prediction, claims, model_key)
    return ContextScore(
        qid=prediction.qid,
        metric="faithfulness",
        score=supported / len(claims),
        n_checked=len(claims),
        n_supported=supported,
    )


async def score_evidence_recall(
    gateway: LLMGateway, query: Query, prediction: Prediction, *, model_key: str | None = None
) -> ContextScore:
    evidence = [e for e in query.gold_evidence if e.strip()]
    if not evidence or not prediction.contexts:
        return _empty(prediction, "evidence_recall")
    supported = await _count_supported(gateway, prediction, evidence, model_key)
    return ContextScore(
        qid=prediction.qid,
        metric="evidence_recall",
        score=supported / len(evidence),
        n_checked=len(evidence),
        n_supported=supported,
    )


async def context_metric_for_run(
    gateway: LLMGateway,
    metric: ContextMetric,
    queries: dict[str, Query],
    predictions: Sequence[Prediction],
    out_path: Path,
    *,
    model_key: str | None = None,
    concurrency: int = 8,
) -> BatchOutcome:
    done = {s.qid for s in read_models(out_path, ContextScore)}
    pending = [p for p in predictions if p.qid in queries and p.qid not in done]

    async def work(prediction: Prediction) -> ContextScore:
        query = queries[prediction.qid]
        if metric == "faithfulness":
            return await score_faithfulness(
                gateway, query.question, prediction, model_key=model_key
            )
        return await score_evidence_recall(gateway, query, prediction, model_key=model_key)

    return await run_resumable(
        pending, work, out_path, concurrency=concurrency, label=lambda p: p.qid
    )
