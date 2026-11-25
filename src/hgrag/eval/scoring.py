"""Turn predictions into per-question scores and aggregated metrics with confidence intervals.

Primary correctness per question:
* unanswerable question → 1 if the pipeline refused, else 0;
* answerable question → the LLM-judge score (1 / 0.5 / 0) when judgments exist, else exact match.

Retrieval is scored at two levels:
* document level: gold documents vs the documents of the retrieved units;
* chunk level: gold chunks (from the index's gold map, Phase 3) vs retrieved chunk IDs.

Per-question scores are written to ``scores.jsonl`` so paired tests can compare systems later.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from hgrag.data.schema import Query
from hgrag.eval.metrics.efficiency import percentile, usd_per_1k_queries
from hgrag.eval.metrics.qa import exact_match, token_f1
from hgrag.eval.metrics.retrieval import hit_at_k, ndcg_at_k, recall_at_k
from hgrag.eval.stats import bootstrap_ci
from hgrag.generation.answers import strip_citations
from hgrag.pipelines.base import Prediction

RECALL_KS = (5, 10)


class RetrievalScore(BaseModel):
    recall: dict[int, float | None]  # None when there is no gold at this level
    hit: dict[int, float | None]  # any gold in top k
    ndcg10: float | None


class QuestionScore(BaseModel):
    qid: str
    query_type: str
    answerable: bool
    correct: float  # primary metric (see module docstring)
    exact_match: float | None  # answerable questions only
    f1: float | None
    judge: float | None
    faithfulness: float | None
    evidence_recall: float | None  # LLM-checked; GraphRAG-Bench retrieval metric
    refused: bool
    doc_retrieval: RetrievalScore | None  # None for pipelines without retrieval
    chunk_retrieval: RetrievalScore | None
    tokens: int
    cost_usd: float
    latency_ms: float


class MetricSummary(BaseModel):
    n: int
    mean: float
    low: float
    high: float


class GroupMetrics(BaseModel):
    n: int
    metrics: dict[str, MetricSummary]
    latency_p50_ms: float
    latency_p95_ms: float
    tokens_per_query: float
    usd_per_1k_queries: float


class RunMetrics(BaseModel):
    primary_metric: str  # "judge" or "exact_match"
    n_missing_predictions: int
    overall: GroupMetrics
    by_query_type: dict[str, GroupMetrics]


def _retrieval(ranked: Sequence[str], gold: Sequence[str]) -> RetrievalScore:
    return RetrievalScore(
        recall={k: recall_at_k(ranked, gold, k) for k in RECALL_KS},
        hit={k: hit_at_k(ranked, gold, k) for k in RECALL_KS},
        ndcg10=ndcg_at_k(ranked, gold, 10),
    )


def score_question(
    query: Query,
    prediction: Prediction,
    judge_score: float | None = None,
    *,
    gold_chunk_ids: Sequence[str] | None = None,
    faithfulness: float | None = None,
    evidence_recall: float | None = None,
) -> QuestionScore:
    answer = strip_citations(prediction.answer)
    contexts = prediction.contexts

    if query.answerable:
        em: float | None = exact_match(answer, query.answers)
        f1: float | None = token_f1(answer, query.answers)
        correct = judge_score if judge_score is not None else em
    else:
        em = f1 = None
        correct = float(prediction.refused)

    doc_ranking = [doc for unit in contexts for doc in unit.source_doc_ids]
    chunk_ranking = [unit.unit_id for unit in contexts if unit.kind == "chunk"]
    return QuestionScore(
        qid=query.qid,
        query_type=query.query_type,
        answerable=query.answerable,
        correct=float(correct or 0.0),
        exact_match=em,
        f1=f1,
        judge=judge_score if query.answerable else None,
        faithfulness=faithfulness,
        evidence_recall=evidence_recall,
        refused=prediction.refused,
        doc_retrieval=_retrieval(doc_ranking, query.gold_doc_ids) if contexts else None,
        chunk_retrieval=(
            _retrieval(chunk_ranking, gold_chunk_ids)
            if contexts and gold_chunk_ids is not None
            else None
        ),
        tokens=prediction.input_tokens + prediction.output_tokens,
        cost_usd=prediction.cost_usd,
        latency_ms=prediction.latency_ms,
    )


def score_run(
    queries: Mapping[str, Query],
    predictions: Sequence[Prediction],
    judge_scores: Mapping[str, float] | None = None,
    *,
    gold_chunks: Mapping[str, Sequence[str]] | None = None,
    faithfulness: Mapping[str, float | None] | None = None,
    evidence_recall: Mapping[str, float | None] | None = None,
) -> tuple[list[QuestionScore], RunMetrics]:
    judge_scores = judge_scores or {}
    faithfulness = faithfulness or {}
    evidence_recall = evidence_recall or {}
    scores = [
        score_question(
            queries[p.qid],
            p,
            judge_scores.get(p.qid),
            gold_chunk_ids=gold_chunks.get(p.qid) if gold_chunks is not None else None,
            faithfulness=faithfulness.get(p.qid),
            evidence_recall=evidence_recall.get(p.qid),
        )
        for p in predictions
        if p.qid in queries
    ]
    by_type: dict[str, list[QuestionScore]] = defaultdict(list)
    for score in scores:
        by_type[score.query_type].append(score)

    answerable = [q for q in queries.values() if q.answerable]
    judged = bool(answerable) and all(q.qid in judge_scores for q in answerable)
    metrics = RunMetrics(
        primary_metric="judge" if judged else "exact_match",
        n_missing_predictions=len(queries) - len(scores),
        overall=_aggregate(scores),
        by_query_type={t: _aggregate(s) for t, s in sorted(by_type.items())},
    )
    return scores, metrics


def _aggregate(scores: Sequence[QuestionScore]) -> GroupMetrics:
    columns: dict[str, list[float]] = {
        "correct": [s.correct for s in scores],
        "exact_match": [s.exact_match for s in scores if s.exact_match is not None],
        "f1": [s.f1 for s in scores if s.f1 is not None],
        "judge": [s.judge for s in scores if s.judge is not None],
        "faithfulness": [s.faithfulness for s in scores if s.faithfulness is not None],
        "evidence_recall": [s.evidence_recall for s in scores if s.evidence_recall is not None],
        "refusal_accuracy": [float(s.refused) for s in scores if not s.answerable],
        "false_refusal_rate": [float(s.refused) for s in scores if s.answerable],
    }
    for level in ("doc", "chunk"):
        retrievals = [r for s in scores if (r := getattr(s, f"{level}_retrieval")) is not None]
        for k in RECALL_KS:
            columns[f"{level}_recall@{k}"] = [
                v for r in retrievals if (v := r.recall[k]) is not None
            ]
            columns[f"{level}_hit@{k}"] = [v for r in retrievals if (v := r.hit[k]) is not None]
        columns[f"{level}_ndcg@10"] = [r.ndcg10 for r in retrievals if r.ndcg10 is not None]

    summaries = {}
    for name, values in columns.items():
        if values:
            interval = bootstrap_ci(values)
            summaries[name] = MetricSummary(
                n=len(values), mean=interval.estimate, low=interval.low, high=interval.high
            )
    latencies = [s.latency_ms for s in scores]
    return GroupMetrics(
        n=len(scores),
        metrics=summaries,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        tokens_per_query=sum(s.tokens for s in scores) / len(scores) if scores else 0.0,
        usd_per_1k_queries=usd_per_1k_queries([s.cost_usd for s in scores]),
    )
