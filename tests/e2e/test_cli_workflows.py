"""CLI workflows on the synthetic corpus: experiments, routers, benchmarks, judges, reports,
and the external-system predictions contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from elevate_rag.cli.main import app
from tests.conftest import offline_config

pytestmark = pytest.mark.e2e


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[CliRunner, Path, Path]:
    runner = CliRunner()
    cfg = offline_config(tmp_path)
    assert runner.invoke(app, ["data", "synthetic", "--data-dir", cfg.data_dir]).exit_code == 0
    pipe = tmp_path / "pipe.yaml"
    pipe.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    spec = tmp_path / "exp.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "id": "C",
                "question": "RQ1",
                "pipeline": "pipe.yaml",
                "datasets": ["synthetic"],
                "variants": [
                    {"name": "V4", "mode": "v4"},
                    {"name": "V1", "mode": "v1"},
                    {"name": "R1", "router": {"level": "rules"}},
                ],
            }
        )
    )
    return runner, tmp_path, spec


def test_experiment_dry_run_run_and_report(workspace: tuple[CliRunner, Path, Path]) -> None:
    runner, tmp, spec = workspace
    runs = tmp / "runs"
    res = runner.invoke(app, ["experiment", "validate", str(spec)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["experiment", "estimate", str(spec)])
    assert res.exit_code == 0 and '"total_usd": 0.0' in res.output
    res = runner.invoke(
        app, ["experiment", "run", str(spec), "--dry-run", "--runs-root", str(runs)]
    )
    assert res.exit_code == 0 and '"plan"' in res.output and not runs.exists()
    res = runner.invoke(app, ["experiment", "run", str(spec), "--runs-root", str(runs)])
    assert res.exit_code == 0, res.output
    assert res.output.count("✓") == 3
    out = tmp / "report"
    res = runner.invoke(app, ["report", "--runs-root", str(runs), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "REPORT.md").exists() and (out / "results.csv").exists()
    assert (out / "paired_comparisons.csv").exists()

    assert not (tmp / "results").exists()  # demo runs never write committed records


GROUSE_ROWS = [
    {  # complete answer -> the judge must say "correct"
        "input": "In which year was Velmora founded?",
        "expected_output": "Velmora was founded in 1712.",
        "actual_output": "It was founded in 1712.",
        "conditions": {"completeness_condition": "==5", "faithfulness_condition": "==1"},
        "metadata": {"test_type": "Highest marks 1"},
    },
    {  # nothing of the reference -> "wrong"
        "input": "In which year was Velmora founded?",
        "expected_output": "Velmora was founded in 1712.",
        "actual_output": "Velmora is a river port.",
        "conditions": {"completeness_condition": "==1", "faithfulness_condition": "==1"},
        "metadata": {"test_type": "Low completeness 1"},
    },
    {  # unfaithful: not applicable to a reference-based correctness judge -> skipped
        "input": "q",
        "expected_output": "a",
        "actual_output": "a plus invented facts",
        "conditions": {"completeness_condition": "==5", "faithfulness_condition": "==0"},
        "metadata": {"test_type": "Low faithfulness 1"},
    },
]


def test_runs_write_traceable_records_with_two_judges(
    workspace: tuple[CliRunner, Path, Path],
) -> None:
    runner, tmp, spec = workspace
    cfg = yaml.safe_load((tmp / "pipe.yaml").read_text())
    cfg["demo"] = False
    cfg["models"]["judge2"] = {"provider": "offline", "model": "judge-b", "family": "offline-b"}
    (tmp / "pipe.yaml").write_text(yaml.safe_dump(cfg))
    results, runs = tmp / "results", tmp / "runs"
    grouse = tmp / "grouse.jsonl"
    grouse.write_text("\n".join(json.dumps(r) for r in GROUSE_ROWS) + "\n")

    res = runner.invoke(
        app,
        [
            "judge",
            "calibrate",
            str(tmp / "pipe.yaml"),
            "--grouse-file",
            str(grouse),
            "--results-dir",
            str(results),
        ],
    )
    assert res.exit_code == 0, res.output
    cals = sorted((results / "judge_calibration").glob("*.json"))
    assert len(cals) == 2  # one file per judge
    assert all(json.loads(p.read_text())["n"] == 2 for p in cals)  # unfaithful case skipped

    res = runner.invoke(
        app,
        ["experiment", "run", str(spec), "--runs-root", str(runs), "--results-dir", str(results)],
    )
    assert res.exit_code == 0, res.output
    records = sorted((results / "runs" / "C").glob("*.json"))
    assert len(records) == 3
    rec = json.loads(records[0].read_text())
    assert rec["config_hash"] and rec["dataset_version"] and rec["code"]["elevate_rag"]
    assert rec["code"]["git_from_run"] is True  # captured in the manifest when the run started
    assert rec["models"]["judge2"]["model"] == "judge-b"
    assert "em" in rec["metrics"] and "correct_judge2" in rec["metrics"]
    assert rec["judge_validation"]["status"] in ("validated", "unvalidated")
    assert set(rec["judge_validation"]["calibration"]) == {"judge", "judge2"}
    assert not rec["raw_run_dir"].startswith("/")  # no absolute paths in committed files
    index = (results / "runs" / "INDEX.csv").read_text().splitlines()
    assert len(index) == 4 and index[0].startswith("experiment,run_id")

    for p in records:  # records can be rebuilt from the raw runs at any time
        p.unlink()
    res = runner.invoke(
        app, ["results", "sync", "--runs-root", str(runs), "--results-dir", str(results)]
    )
    assert res.exit_code == 0 and "3 run record(s)" in res.output
    assert len(list((results / "runs" / "C").glob("*.json"))) == 3


def test_router_cli_and_benchmarks(workspace: tuple[CliRunner, Path, Path]) -> None:
    runner, tmp, _ = workspace
    data = str(tmp / "data")
    res = runner.invoke(
        app,
        [
            "router",
            "train",
            str(tmp / "r2"),
            "--dataset",
            "synthetic",
            "--split",
            "dev",
            "--data-dir",
            data,
        ],
    )
    assert res.exit_code == 0, res.output
    assert json.loads((tmp / "r2" / "meta.json").read_text())["scheme"] == "type"
    res = runner.invoke(app, ["router", "cv", "--dataset", "synthetic", "--data-dir", data])
    # dev split of the 15-question fixture is too small for 5 stratified folds → loud failure
    assert res.exit_code != 0
    pipe = tmp / "pipe.yaml"
    assert runner.invoke(app, ["index", "build", str(pipe)]).exit_code == 0
    res = runner.invoke(
        app,
        [
            "bench",
            "vector",
            str(pipe),
            "--backends",
            "exact,faiss",
            "--n-queries",
            "10",
            "--out",
            str(tmp / "vb.json"),
        ],
    )
    assert res.exit_code == 0, res.output
    bench = json.loads((tmp / "vb.json").read_text())
    assert [b["backend"] for b in bench] == ["exact", "faiss"]
    res = runner.invoke(app, ["bench", "graph", str(pipe), "--out", str(tmp / "gb.json")])
    assert res.exit_code == 0, res.output


def test_external_predictions_contract(workspace: tuple[CliRunner, Path, Path]) -> None:
    runner, tmp, _ = workspace
    from elevate_rag.data.prepare import DatasetStore

    qs = DatasetStore.open(tmp / "data", "synthetic").questions("test")
    preds = tmp / "preds.jsonl"
    with preds.open("w") as fh:
        for q in qs[:-1]:  # one prediction deliberately missing
            fh.write(
                json.dumps(
                    {
                        "question_id": q.question_id,
                        "answer": q.answers[0] if q.answers else "insufficient information",
                        "contexts": ["some text"],
                        "prompt_tokens": 50,
                    }
                )
                + "\n"
            )
    res = runner.invoke(
        app,
        [
            "experiment",
            "external",
            str(preds),
            "--system",
            "hipporag2",
            "--dataset",
            "synthetic",
            "--data-dir",
            str(tmp / "data"),
            "--runs-root",
            str(tmp / "runs"),
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(next((tmp / "runs" / "EXT").rglob("summary.json")).read_text())
    assert summary["missing_predictions"] == 1 and summary["error_rate"] > 0
    bad = tmp / "bad.jsonl"
    bad.write_text('{"question_id": "nope", "answer": "x"}\n')
    res = runner.invoke(
        app,
        [
            "experiment",
            "external",
            str(bad),
            "--system",
            "x",
            "--dataset",
            "synthetic",
            "--data-dir",
            str(tmp / "data"),
        ],
    )
    assert res.exit_code != 0
