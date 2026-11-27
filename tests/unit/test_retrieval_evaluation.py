import math

import numpy as np
import pytest

from elevate_rag.data import CharacterTokenizer, Chunk, Evidence
from elevate_rag.evaluation import audit_agreement, coverage, normalize, retrieval_metrics, token_f1
from elevate_rag.pipeline import assemble, parse_answer
from elevate_rag.retrieval import BM25, ExactIndex, Hit, rrf
from elevate_rag.statistics import holm, interval, mcnemar, minimum_detectable_effect, paired


def corpus():
    return {
        "b": Chunk("b", "doc1", "Tree", "The bright beacon in a tree.", "wiki", 0, 28),
        "a": Chunk("a", "doc1", "Tree", "The bright beacon in a tree.", "wiki", 34, 62),
        "c": Chunk("c", "doc2", "Far off", "Nothing here.", "other", 0, 13),
    }


def test_sparse_dense_and_rrf():
    chunks = list(corpus().values())
    bm25 = BM25(chunks)
    assert [hit.chunk_id for hit in bm25.search("bright beacon", 5)] == ["a", "b"]
    assert not bm25.search("unknown", 5)
    dense = ExactIndex(2)
    dense.upsert(chunks, np.array([[0, 1], [1, 0], [-1, 0]], np.float32))
    assert [hit.chunk_id for hit in dense.search(np.array([1, 0]), 5, source="wiki")] == ["a", "b"]
    with pytest.raises(ValueError, match="expected"):
        dense.search(np.array([1, 0, 0]), 5)
    scores = rrf([[Hit("b", 9), Hit("b", 8), Hit("a", 7)], [Hit("a", 50)]], 60, 5)
    assert [hit.chunk_id for hit in scores] == ["a", "b"]
    assert scores[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_context_cap_and_citations():
    catalog = corpus()
    ctx = assemble(
        "Who?",
        [Hit("a", 2), Hit("b", 1)],
        catalog,
        CharacterTokenizer(),
        context_budget=75,
        context_window=2000,
        output_limit=20,
    )
    assert len(ctx.passages) == 1
    assert ctx.context_tokens <= 75
    assert parse_answer("A beacon [1].", ctx, closed_book=False).citations[0].chunk_id == "a"
    for invalid in ("A beacon [2].", "a beacon", "insufficient information [1]"):
        with pytest.raises(ValueError):
            parse_answer(invalid, ctx, closed_book=False)
    assert parse_answer("Insufficient information.", ctx, closed_book=False).refused
    assert not parse_answer("A beacon", ctx, closed_book=True).citations
    with pytest.raises(ValueError, match="context window"):
        assemble(
            "Who?",
            [],
            catalog,
            CharacterTokenizer(),
            context_budget=10,
            context_window=100,
            output_limit=10,
        )


def test_evidence_metrics_and_exact_match():
    chunks = corpus()
    support = (Evidence("doc1", 5, 9), Evidence("doc2"))
    assert coverage(support, [chunks["a"], chunks["c"]]) == 0.5
    values = retrieval_metrics(
        support, [Hit("c", 9), Hit("a", 8), Hit("b", 7)], chunks, [chunks["b"]]
    )
    assert values["recall_at_10"] == 1
    assert values["gold_context_precision"] == 1
    assert values["gold_context_recall"] == 0.5
    assert values["ndcg_at_10"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert retrieval_metrics([], [], chunks, [])["recall_at_10"] is None
    assert normalize("The Beacon [1].") == "beacon"
    assert token_f1("the north beacon", "north marker") == 0.5
    assert token_f1("not yes", "yes") == 0


def test_paired_statistics_and_audit():
    assert interval([0.8] * 4)["estimate"] == pytest.approx(0.8)
    assert interval([])["lower"] is None
    assert paired({"q": 0.5}, {"q": 0.75})["estimate"] == 0.25
    with pytest.raises(ValueError, match="aligned"):
        paired({"q": 0.0}, {"x": 0.0})
    left = {str(i): True for i in range(5)}
    right = dict.fromkeys(left, False)
    assert mcnemar(left, right) == {"left_only": 5, "right_only": 0, "p_value": 0.0625}
    assert minimum_detectable_effect(left, right, 500)["approximate_mde"] > 0
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    output = audit_agreement(
        [
            {"annotator_a": "correct", "annotator_b": "correct", "judge_grade": "correct"},
            {"annotator_a": "wrong", "annotator_b": "wrong", "judge_grade": "wrong"},
            {"annotator_a": None, "annotator_b": None},
        ]
    )
    assert output["double_annotated"] == 2
    assert output["human_human_kappa"] == 1
