import math
from typing import Any

from hgrag.data.schema import Query
from hgrag.eval.scoring import score_question, score_run
from hgrag.pipelines.base import ContextUnit, Prediction


def _query(qid: str, answerable: bool = True, **kwargs: Any) -> Query:
    return Query(
        qid=qid,
        question="q?",
        answers=["Walt Disney"] if answerable else [],
        gold_doc_ids=kwargs.pop("gold", ["d1"] if answerable else []),
        query_type="multi_hop" if answerable else "unanswerable",
        native_type="x",
        answerable=answerable,
        **kwargs,
    )


def _prediction(qid: str, answer: str, refused: bool = False, **kwargs: Any) -> Prediction:
    defaults: dict[str, Any] = {
        "model_key": "m",
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": 0.001,
        "latency_ms": 100.0,
        "cache_hit": False,
        "status": "ok",
    }
    return Prediction(qid=qid, answer=answer, refused=refused, **{**defaults, **kwargs})


def test_answerable_uses_exact_match_without_judge() -> None:
    score = score_question(_query("q1"), _prediction("q1", "Walt Disney [2]"))
    assert score.correct == 1.0 and score.exact_match == 1.0 and score.f1 == 1.0
    assert score.doc_retrieval is None  # no contexts → retrieval metrics undefined


def test_judge_score_overrides_exact_match() -> None:
    score = score_question(_query("q1"), _prediction("q1", "Disney, Walt"), judge_score=0.5)
    assert score.correct == 0.5 and score.exact_match == 0.0


def test_unanswerable_is_correct_only_when_refused() -> None:
    query = _query("u1", answerable=False)
    assert score_question(query, _prediction("u1", "insufficient information", True)).correct == 1
    wrong = score_question(query, _prediction("u1", "Paris"))
    assert wrong.correct == 0 and wrong.exact_match is None


def test_doc_level_recall_from_contexts() -> None:
    contexts = [
        ContextUnit(unit_id="c1", kind="chunk", text="t", source_doc_ids=["d9"]),
        ContextUnit(unit_id="c2", kind="chunk", text="t", source_doc_ids=["d1"]),
    ]
    score = score_question(
        _query("q1", gold=["d1", "d2"]),
        _prediction("q1", "x", contexts=contexts),
        gold_chunk_ids=["c2"],
    )
    assert score.doc_retrieval is not None and score.doc_retrieval.recall[5] == 0.5
    assert score.chunk_retrieval is not None and score.chunk_retrieval.recall[5] == 1.0
    assert score.chunk_retrieval.ndcg10 == 1 / math.log2(3)  # gold chunk at rank 2


def test_score_run_aggregates_by_type_and_counts_missing() -> None:
    queries = {q.qid: q for q in (_query("q1"), _query("q2"), _query("u1", answerable=False))}
    predictions = [
        _prediction("q1", "Walt Disney"),
        _prediction("u1", "insufficient information", True),
    ]
    _, metrics = score_run(queries, predictions)

    assert metrics.primary_metric == "exact_match"
    assert metrics.n_missing_predictions == 1
    assert metrics.overall.n == 2 and metrics.overall.metrics["correct"].mean == 1.0
    assert metrics.by_query_type["unanswerable"].metrics["refusal_accuracy"].mean == 1.0
    assert metrics.overall.usd_per_1k_queries == 1.0


def test_primary_metric_is_judge_when_all_answerable_are_judged() -> None:
    queries = {"q1": _query("q1")}
    _, metrics = score_run(queries, [_prediction("q1", "wrong")], judge_scores={"q1": 1.0})
    assert metrics.primary_metric == "judge" and metrics.overall.metrics["correct"].mean == 1.0
