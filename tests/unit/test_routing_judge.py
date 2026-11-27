from __future__ import annotations

from pathlib import Path

import pytest

from elevate_rag.config import RouterConfig
from elevate_rag.evaluation.judge import CorrectnessJudge, PairwiseJudge, parse_json_object
from elevate_rag.pipeline import QueryRecord
from elevate_rag.routing.fallback import should_escalate, tune_threshold
from elevate_rag.routing.oracle import oracle_labels, outcome_training_labels, routing_report
from elevate_rag.routing.router import (
    Router,
    TfidfSvmRouter,
    classify_rules,
    cross_validate,
    type_labels_valid,
)
from elevate_rag.utils import ConfigError


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("What are the main themes of the novel?", "global"),
        ("Which company has more employees, Lumenward or Corvane Systems?", "comparison"),
        ("Did the Tidal Act pass before Mirela Voss left office?", "temporal"),
        ("Where was the founder of Lumenward born?", "multi_hop"),
        ("When was Velmora founded?", "single_hop"),
    ],
)
def test_rule_router(q: str, expected: str) -> None:
    assert classify_rules(q)[0] == expected


async def test_router_levels_map_types_to_modes(scripted) -> None:
    cfg = RouterConfig(level="rules")
    d = await Router(cfg).decide("Where was the founder of Lumenward born?")
    assert (d.level, d.query_type, d.mode) == ("R1", "multi_hop", "local") and d.reasons
    static = await Router(RouterConfig(level="static", static_mode="mix")).decide("x")
    assert static.mode == "mix"
    llm = scripted(['{"query_type": "global"}', "garbage", '{"query_type": "weird"}'])
    r3 = Router(RouterConfig(level="llm"), llm=llm)
    assert (await r3.decide("q")).mode == "global"
    fb = await r3.decide("When was Velmora founded?")
    assert fb.query_type == "single_hop" and "fell back" in fb.reasons[0]
    assert "invalid LLM label" in (await r3.decide("When?")).reasons[0]
    with pytest.raises(ConfigError):
        Router(RouterConfig(level="classifier"))
    with pytest.raises(ConfigError):
        Router(RouterConfig(level="llm"))


def _toy() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for i in range(30):
        texts += [
            f"When was town {i} founded?",
            f"Which is older, town {i} or city {i}?",
            f"What are the main themes across region {i}?",
        ]
        labels += ["single_hop", "comparison", "global"]
    return texts, labels


def test_tfidf_svm_router_fit_persist_and_cv(tmp_path: Path) -> None:
    texts, labels = _toy()
    m = TfidfSvmRouter("type").fit(texts, labels)
    assert m.predict(["Which is older, A or B?"]) == ["comparison"]
    m.save(tmp_path / "r")
    loaded = TfidfSvmRouter.load(tmp_path / "r")
    assert loaded.predict_one("What are the main themes across X?")[0] == "global"
    assert loaded.meta["n_train"] == 90 and loaded.meta["scheme"] == "type"
    cv = cross_validate(texts, labels, folds=5)
    assert cv["accuracy"] > 0.9 and len(cv["folds"]) == 5
    groups = ["a"] * 45 + ["b"] * 45
    lodo = cross_validate(texts, labels, groups=groups)
    assert lodo["protocol"] == "leave-one-dataset-out" and {
        f["held_out"] for f in lodo["folds"]
    } == {"a", "b"}
    with pytest.raises(ValueError):
        TfidfSvmRouter().fit(["a", "b"], ["x", "x"])
    with pytest.raises(ConfigError):
        TfidfSvmRouter("nope")
    with pytest.raises(ConfigError):
        type_labels_valid(["single_hop", "bogus"])


def test_tune_threshold_maximises_dev_accuracy() -> None:
    scores = [0.9, 0.8, 0.3, 0.2, None]
    cheap = [True, True, False, False, False]
    esc = [True, False, True, True, True]
    out = tune_threshold(scores, cheap, esc)
    assert out["dev_accuracy"] == 1.0 and 0.3 < out["threshold"] <= 0.8
    assert out["escalation_rate"] == pytest.approx(3 / 5)
    with pytest.raises(ValueError):
        tune_threshold([0.1], [True], [])


def test_should_escalate_reasons() -> None:
    cfg = RouterConfig(level="fallback", fallback_threshold=0.5)
    ok = QueryRecord("q", "q", "v4", top_score=0.9)
    assert should_escalate(ok, cfg) is None
    assert "threshold" in should_escalate(QueryRecord("q", "q", "v4", top_score=0.1), cfg)
    assert "insufficient" in should_escalate(
        QueryRecord("q", "q", "v4", refused=True, top_score=0.9), cfg
    )
    assert should_escalate(QueryRecord("q", "q", "v4", error="x"), cfg) == "initial path failed"


def test_oracle_and_routing_report() -> None:
    per_mode = {
        "v4": {
            "a": {"correct": True, "tokens": 100},
            "b": {"correct": False, "tokens": 100},
            "c": {"correct": False, "tokens": 100},
        },
        "mix": {
            "a": {"correct": True, "tokens": 900},
            "b": {"correct": True, "tokens": 900},
            "c": {"correct": False, "tokens": 900},
        },
    }
    assert oracle_labels(per_mode) == {"a": "v4", "b": "mix", "c": None}
    assert outcome_training_labels(per_mode)["c"] == "v4"  # nobody right → cheapest
    routed = {
        "a": {"mode": "v4", "correct": True, "tokens": 100, "cost_usd": 0.0},
        "b": {"mode": "v4", "correct": False, "tokens": 100, "cost_usd": 0.0},
        "c": {"mode": "mix", "correct": False, "tokens": 900, "cost_usd": 0.0},
    }
    rep = routing_report(per_mode, routed, reference_mode="mix")
    assert rep["oracle_accuracy"] == pytest.approx(2 / 3)
    assert rep["regret"] == pytest.approx(1 / 3)
    assert rep["best_static_mode"] == "mix" and rep["routing_accuracy_vs_oracle"] == 0.5
    assert rep["token_savings_vs_reference"] == pytest.approx(1 - 1100 / 2700)
    with pytest.raises(ValueError):
        routing_report(per_mode, {"zzz": routed["a"]})


# ------------------------------------------------------------------------------- judge


async def test_correctness_judge(scripted) -> None:
    llm = scripted(['```json\n{"verdict": "Correct", "reason": "same"}\n```', "nope", "still nope"])
    j = CorrectnessJudge(llm)
    v = await j.judge("q", ["Kessering"], "It is Kessering", answerable=True, refused=False)
    assert v.verdict == "correct" and v.score == 1.0
    bad = await j.judge("q", ["x"], "y", answerable=True, refused=False)
    assert bad.verdict == "invalid" and bad.score is None  # missing, never "wrong"
    assert len(llm.prompts) == 3  # one retry for the unparseable answer
    un = await j.judge("q", [], "insufficient information", answerable=False, refused=True)
    assert un.verdict == "correct" and un.judged_by == "rule"
    halluc = await j.judge("q", [], "Paris", answerable=False, refused=False)
    assert halluc.verdict == "wrong"
    ref = await j.judge("q", ["x"], "insufficient information", answerable=True, refused=True)
    assert ref.verdict == "wrong" and len(llm.prompts) == 3  # rule, no LLM call


async def test_pairwise_judge_requires_agreement_in_both_orders(scripted) -> None:
    agree = scripted(
        [
            '{"comprehensiveness":"A","diversity":"B","empowerment":"A","overall":"A"}',
            '{"comprehensiveness":"B","diversity":"B","empowerment":"tie","overall":"B"}',
        ]
    )
    out = await PairwiseJudge(agree).compare("q", "ans1", "ans2")
    assert out == {
        "comprehensiveness": "1",
        "diversity": "tie",
        "empowerment": "tie",
        "overall": "1",
    }
    broken = await PairwiseJudge(scripted(["x", "y"])).compare("q", "a", "b")
    assert set(broken.values()) == {"invalid"}


def test_parse_json_object() -> None:
    assert parse_json_object('noise {"a": 1} tail') == {"a": 1}
    assert parse_json_object("[1, 2]") is None and parse_json_object("{bad") is None


async def test_compare_runs_only_global_questions(scripted) -> None:
    from elevate_rag.evaluation.pairwise import compare_runs
    from elevate_rag.types import Question

    qs = [
        Question(question_id="g", dataset="d", question="themes?", query_type="global"),
        Question(question_id="s", dataset="d", question="when?", query_type="single_hop"),
    ]
    a = [{"question_id": "g", "answer": "long"}, {"question_id": "s", "answer": "x"}]
    b = [{"question_id": "g", "answer": "short"}, {"question_id": "s", "answer": "y"}]
    llm = scripted(
        [
            '{"comprehensiveness":"A","diversity":"A","empowerment":"B","overall":"A"}',
            '{"comprehensiveness":"B","diversity":"A","empowerment":"A","overall":"B"}',
        ]
    )
    res = await compare_runs(a, b, qs, llm)
    assert res["n"] == 1 and len(llm.prompts) == 2  # single-hop question not judged
    assert res["comprehensiveness"]["a_wins"] == 1.0 and res["overall"]["a_wins"] == 1.0
    assert res["diversity"]["ties"] == 1.0 and res["empowerment"]["b_wins"] == 1.0
