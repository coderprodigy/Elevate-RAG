"""CLI workflows on the synthetic corpus: experiments, routers, benchmarks, annotation, reports,
and the external-system predictions contract."""

from __future__ import annotations

import csv
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

    run_dir = next((runs / "C").glob("C-V4-*"))
    res = runner.invoke(
        app,
        ["annotate", "failures", str(run_dir), str(tmp / "f.csv"), "--data-dir", str(tmp / "data")],
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        [
            "annotate",
            "judge-sample",
            str(run_dir),
            str(tmp / "j.csv"),
            "--data-dir",
            str(tmp / "data"),
        ],
    )
    assert res.exit_code == 0 and (tmp / "j.csv").exists()


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


def test_triple_agreement_cli(tmp_path: Path) -> None:
    for name, labels in (
        ("a", ["correct", "wrong", "partial"]),
        ("b", ["correct", "wrong", "wrong"]),
    ):
        with (tmp_path / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["triple_id", "label"])
            w.writeheader()
            w.writerows({"triple_id": f"t{i}", "label": lab} for i, lab in enumerate(labels))
    res = CliRunner().invoke(
        app, ["annotate", "triple-agreement", str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]
    )
    assert res.exit_code == 0 and '"kappa"' in res.output
