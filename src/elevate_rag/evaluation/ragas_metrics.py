"""RAG metrics from the `ragas` package (Es et al.; the de-facto standard RAG metric suite).

Scored with the judge model over an OpenAI-compatible endpoint (vLLM / Ollama). The package
makes its own LLM calls, which bypass this project's cost ledger and response cache, so only a
$0 self-hosted judge is accepted (checked in `experiments.runner.validate`). For a paid judge use
the native implementation of the same definitions (`llm_metrics`, spec `llm_metrics: true`).

Metrics added to each scored row (None = undefined, e.g. a refusal has no claims):

* ``ragas_faithfulness``        claims in the answer supported by the retrieved context
* ``ragas_context_precision``   retrieved passages useful for the reference answer, rank-weighted
* ``ragas_context_recall``      reference-answer statements attributable to the context
* ``ragas_answer_correctness``  factual overlap of answer and reference (LLM-only, weight 1/0)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Any

from elevate_rag.config import ProviderConfig, require_resolved
from elevate_rag.evaluation.harness import context_texts
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.types import Question
from elevate_rag.utils import ConfigError

log = logging.getLogger(__name__)

RAGAS_KEYS = (
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_correctness",
)


def make_ragas_metrics(cfg: ProviderConfig) -> tuple[dict[str, Any], Any]:
    """Build the four ragas metric objects on the judge endpoint. Returns (metrics, client); close
    the client (`await client.close()`) inside the same event loop when scoring is done."""
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("ragas metrics need the extra: uv sync --extra ragas") from exc
    if cfg.provider != "openai-compatible":
        raise ConfigError(f"ragas needs an openai-compatible judge, got {cfg.provider!r}")
    base = require_resolved(cfg.base_url, cfg.model) or os.getenv("OPENAI_COMPAT_BASE_URL")
    if not base:
        raise ConfigError(f"{cfg.model}: base_url is required for ragas")
    client = AsyncOpenAI(
        base_url=base.rstrip("/"),
        api_key=(os.getenv(cfg.api_key_env) if cfg.api_key_env else None) or "not-needed",
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )
    llm = llm_factory(cfg.model, client=client, max_tokens=2048)
    metrics = {
        "ragas_faithfulness": Faithfulness(llm=llm),
        "ragas_context_precision": ContextPrecision(llm=llm),
        "ragas_context_recall": ContextRecall(llm=llm),
        "ragas_answer_correctness": AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
    }
    return metrics, client


async def _value(coro: Any) -> float | None:
    try:
        res = await coro
    except Exception as exc:  # a failed metric is missing, never zero
        log.warning("ragas metric failed: %s: %s", type(exc).__name__, str(exc)[:200])
        return None
    v = getattr(res, "value", res)
    return float(v) if isinstance(v, int | float) else None


async def add_ragas_scores(
    scored: Sequence[dict[str, Any]],
    questions: Sequence[Question],
    store: ChunkStore | None,
    metrics: dict[str, Any],
    *,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Return the rows with the RAGAS metrics added to `scores` (answerable questions only)."""
    qmap = {q.question_id: q for q in questions}
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        s = dict(row["scores"])
        q = qmap[row["question_id"]]
        out: dict[str, float | None] = dict.fromkeys(RAGAS_KEYS)
        answer = "" if row.get("error") else str(row.get("answer") or "")
        refused = bool(row.get("refused"))
        texts = context_texts(row, store)
        ref = q.answers[0] if q.answers else ""
        if q.answerable and ref:
            async with sem:
                if texts:
                    out["ragas_context_precision"] = await _value(
                        metrics["ragas_context_precision"].ascore(
                            user_input=q.question, reference=ref, retrieved_contexts=texts
                        )
                    )
                    out["ragas_context_recall"] = await _value(
                        metrics["ragas_context_recall"].ascore(
                            user_input=q.question, retrieved_contexts=texts, reference=ref
                        )
                    )
                if answer.strip() and not refused:
                    if texts:
                        out["ragas_faithfulness"] = await _value(
                            metrics["ragas_faithfulness"].ascore(
                                user_input=q.question, response=answer, retrieved_contexts=texts
                            )
                        )
                    out["ragas_answer_correctness"] = await _value(
                        metrics["ragas_answer_correctness"].ascore(
                            user_input=q.question, response=answer, reference=ref
                        )
                    )
        s.update(out)
        return {**row, "scores": s}

    return list(await asyncio.gather(*(one(r) for r in scored)))
