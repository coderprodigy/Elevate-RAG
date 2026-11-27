"""Automated judge validation (GroUSE calibration + inter-judge agreement), RAGAS scoring and the
committed run records (D-36)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from elevate_rag.config import ProviderConfig
from elevate_rag.evaluation.harness import judge_agreement, summarize
from elevate_rag.evaluation.judge_calibration import (
    CALIBRATION_MIN,
    CalibrationCase,
    calibrate,
    calibration_key,
    expected_verdict,
    judge_validation,
    load_calibration,
    load_cases,
    passes,
    revalidate,
    write_calibration,
)
from elevate_rag.evaluation.ragas_metrics import RAGAS_KEYS, add_ragas_scores
from elevate_rag.providers.base import UsageLedger
from elevate_rag.providers.llm import LLMResult, OfflineLLM
from elevate_rag.tracking import build_record, git_info, rebuild_index, write_record
from elevate_rag.types import Question

# ------------------------------------------------------------------------------- calibration


def test_grouse_conditions_map_to_correctness_verdicts() -> None:
    assert expected_verdict({"completeness_condition": "==5"}) == "correct"
    assert expected_verdict({"completeness_condition": "<5"}) == "not_correct"
    assert expected_verdict({"completeness_condition": "==1"}) == "wrong"
    assert expected_verdict({"completeness_condition": "==None"}) is None  # refusal cases
    unfaithful = {"completeness_condition": "==5", "faithfulness_condition": "==0"}
    assert expected_verdict(unfaithful) is None  # not a reference-correctness question
    assert passes("not_correct", "partial") and passes("not_correct", "wrong")
    assert not passes("correct", "partial") and not passes("wrong", "partial")


def test_load_cases_skips_inapplicable_tests(tmp_path: Path) -> None:
    rows = [
        {
            "input": "q",
            "expected_output": "ref",
            "actual_output": "ans",
            "conditions": {"completeness_condition": c, "faithfulness_condition": "==1"},
            "metadata": {"test_type": f"Type {i}"},
        }
        for i, c in enumerate(["==5", "==None", "<5"])
    ]
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cases = load_cases(p)
    assert [c.expected for c in cases] == ["correct", "not_correct"]
    assert cases[0].test_type == "Type"  # trailing set number stripped for per-type stats


class _FixedJudge(OfflineLLM):
    def __init__(self, verdict: str) -> None:
        super().__init__(
            ProviderConfig(provider="offline", model="fixed", revision="r1"),
            role="judge",
            ledger=UsageLedger(),
        )
        self.verdict = verdict

    async def _call(self, messages: Any, system: Any, max_tokens: int, hints: Any) -> LLMResult:
        return LLMResult(
            text=json.dumps({"verdict": self.verdict, "reason": "fixed"}),
            model="fixed",
            prompt_tokens=1,
            completion_tokens=1,
            latency_s=0.0,
        )


async def test_calibrate_scores_pass_rate_and_round_trips(tmp_path: Path) -> None:
    cases = [
        CalibrationCase("c1", "Highest marks", "q", "ref", "ans", "correct"),
        CalibrationCase("c2", "Highest marks", "q", "ref", "ans", "correct"),
        CalibrationCase("c3", "Low completeness", "q", "ref", "ans", "wrong"),
    ]
    res = await calibrate(_FixedJudge("correct"), cases)
    assert res["n"] == 3 and res["passed"] == 2 and res["pass_rate"] == pytest.approx(2 / 3)
    assert not res["calibrated"]  # 0.67 < 0.8
    assert res["by_test_type"]["Low completeness"]["pass_rate"] == 0.0
    write_calibration(tmp_path, res)
    assert load_calibration(tmp_path, "fixed", "r1") == res
    assert load_calibration(tmp_path, "fixed", "other-revision") is None
    assert "/" not in calibration_key("org/model", "abc")


def test_judge_validation_requires_both_calibrations_and_kappa() -> None:
    good = {"pass_rate": 0.9, "n": 63, "calibrated": True}
    bad = {"pass_rate": 0.5, "n": 63, "calibrated": False}
    agree = {"n": 40, "kappa": 0.7}
    ok = judge_validation(agree, {"judge": good, "judge2": good})
    assert ok["status"] == "validated" and ok["reasons"] == []
    assert judge_validation({"n": 40, "kappa": 0.4}, {"judge": good, "judge2": good})["status"] == (
        "unvalidated"
    )
    r = judge_validation(agree, {"judge": good, "judge2": bad})
    assert r["status"] == "unvalidated" and str(CALIBRATION_MIN) in r["reasons"][0]
    r = judge_validation(None, {"judge": good})  # one judge only: agreement unmeasurable
    assert r["status"] == "unvalidated" and any("second judge" in x for x in r["reasons"])
    r = judge_validation(agree, {"judge": None, "judge2": good})
    assert any("not calibrated" in x for x in r["reasons"])


# ------------------------------------------------------------------------------- agreement


def _scored(v1: str, v2: str, rule: bool = False) -> dict[str, Any]:
    return {
        "question_id": f"q{v1}{v2}{rule}",
        "scores": {
            "query_type": "single_hop",
            "answerable": True,
            "failed": False,
            "judge": v1,
            "judge2": v2,
            "judge_rule": rule,
            "correct": v1 == "correct",
            "correct_judge2": v2 == "correct",
        },
    }


def test_judge_agreement_ignores_rule_verdicts_and_invalid() -> None:
    rows = [
        _scored("correct", "correct"),
        _scored("wrong", "wrong"),
        _scored("correct", "wrong"),
        _scored("wrong", "wrong", rule=True),  # refusal rule: excluded
        _scored("invalid", "correct"),  # unparseable: excluded
    ]
    a = judge_agreement(rows)
    assert a is not None and a["n"] == 3
    assert a["raw_agreement"] == pytest.approx(2 / 3)
    assert a["kappa"] == pytest.approx(0.4)  # po 2/3, pe 4/9
    assert judge_agreement([{"question_id": "x", "scores": {"judge": "correct"}}]) is None
    s = summarize(rows)
    assert s["judge_agreement"]["n"] == 3 and "judge_rule" not in s["overall"]


# ------------------------------------------------------------------------------- ragas


class _Metric:
    def __init__(self, value: float | Exception) -> None:
        self.value, self.calls = value, 0

    async def ascore(self, **kw: Any) -> Any:
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return type("R", (), {"value": self.value})()


async def test_ragas_scores_answerable_rows_and_records_failures_as_missing() -> None:
    qs = [
        Question(question_id="a", dataset="d", question="q?", answers=["x"]),
        Question(question_id="u", dataset="d", question="q?", answers=[], answerable=False),
    ]
    rows = [
        {"question_id": "a", "answer": "x", "context_texts": ["ctx"], "scores": {"em": 1.0}},
        {"question_id": "u", "answer": "", "refused": True, "context_texts": [], "scores": {}},
    ]
    metrics = {
        "ragas_faithfulness": _Metric(0.75),
        "ragas_context_precision": _Metric(1.0),
        "ragas_context_recall": _Metric(RuntimeError("server down")),
        "ragas_answer_correctness": _Metric(0.5),
    }
    out = await add_ragas_scores(rows, qs, None, metrics)
    a, u = out[0]["scores"], out[1]["scores"]
    assert a["ragas_faithfulness"] == 0.75 and a["ragas_context_precision"] == 1.0
    assert a["ragas_context_recall"] is None  # a failed metric is missing, never zero
    assert a["ragas_answer_correctness"] == 0.5 and a["em"] == 1.0
    assert all(u[k] is None for k in RAGAS_KEYS)  # unanswerable: nothing to score
    assert metrics["ragas_faithfulness"].calls == 1


# ------------------------------------------------------------------------------- records


def _manifest() -> dict[str, Any]:
    return {
        "definition": {
            "run_id": "E-V4-ds-abc",
            "experiment": "E",
            "research_question": "RQ1",
            "dataset": "ds",
            "variant": {"name": "V4", "mode": "v4"},
            "split": "test",
            "limit": None,
            "config_hash": "h1",
            "index_key": "k1",
            "config": {"models": {"generator": {"provider": "p", "model": "m", "revision": "r"}}},
        },
        "created_at": "2026-11-27T00:00:00+00:00",
        "updated_at": "2026-11-27T01:00:00+00:00",
        "environment": {"python": "3.12"},
        "status": "complete",
    }


def test_run_record_is_traceable_and_index_rebuilds(tmp_path: Path) -> None:
    summary = {
        "overall": {"n": 20, "em": {"mean": 0.3, "ci_low": 0.1, "ci_high": 0.5, "n": 20}},
        "dataset_version": "dv1",
        "judge_agreement": {"n": 18, "kappa": 0.7},
        "judge_validation": {"status": "validated"},
        "ledger": {"spent_usd": 0.0},
    }
    git = {"commit": "c" * 40, "branch": "main", "dirty": False}
    rec = build_record(_manifest(), summary, tmp_path / "runs" / "E" / "x", git)
    assert rec["metrics"]["em"]["mean"] == 0.3 and rec["dataset_version"] == "dv1"
    assert rec["code"]["git_commit"] == "c" * 40 and rec["cost_usd"] == 0.0
    assert rec["code"]["git_from_run"] is False  # manifest predates git capture
    m = _manifest()
    m["environment"]["git"] = {"commit": "r" * 40, "branch": "b", "dirty": False}
    assert build_record(m, summary, tmp_path, git)["code"]["git_commit"] == "r" * 40
    assert rec["models"]["generator"] == {
        "provider": "p",
        "model": "m",
        "revision": "r",
        "family": None,
    }
    assert rec["raw_run_dir"] == "E/E-V4-ds-abc" and "host" not in rec["environment"]
    path = write_record(tmp_path / "results", rec)
    assert path == tmp_path / "results" / "runs" / "E" / "E-V4-ds-abc.json"
    rows = list(csv.DictReader((tmp_path / "results" / "runs" / "INDEX.csv").open()))
    assert rows[0]["em"] == "0.3" and rows[0]["judge_status"] == "validated"
    assert rows[0]["git_commit"] == "c" * 12
    path.unlink()
    assert list(csv.DictReader(rebuild_index(tmp_path / "results").open())) == []


def test_git_info_outside_a_repository(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"commit": None, "branch": None, "dirty": None}
    info = git_info(Path(__file__).parent)
    assert info["commit"] is None or len(info["commit"]) == 40


def test_mlflow_logging_round_trip(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from elevate_rag.tracking import log_mlflow

    summary = {"overall": {"n": 2, "f1": {"mean": 0.5, "ci_low": 0.2, "ci_high": 0.8, "n": 2}}}
    rec = build_record(_manifest(), summary, tmp_path, {"commit": "abc", "dirty": True})
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_mlflow(rec, tmp_path, uri)
    mlflow.set_tracking_uri(uri)
    got = mlflow.get_run(run_id)
    assert got.data.metrics["f1.mean"] == 0.5 and got.data.tags["git_commit"] == "abc"
    assert got.data.params["dataset"] == "ds"


def test_revalidate_uses_calibrations_written_after_the_run(tmp_path: Path) -> None:
    config = {
        "models": {
            "judge": {"model": "j1", "revision": "a"},
            "judge2": {"model": "j2", "revision": None},
        }
    }
    summary = {"judge_agreement": {"n": 30, "kappa": 0.8}, "judge_validation": {"status": "x"}}
    assert revalidate(summary, config, tmp_path)["status"] == "unvalidated"  # no files yet
    for model, rev in (("j1", "a"), ("j2", None)):
        write_calibration(
            tmp_path,
            {"judge": calibration_key(model, rev), "pass_rate": 0.9, "n": 63, "calibrated": True},
        )
    assert revalidate(summary, config, tmp_path)["status"] == "validated"
    assert revalidate({}, {"models": {}}, tmp_path) is None  # run without any judge
