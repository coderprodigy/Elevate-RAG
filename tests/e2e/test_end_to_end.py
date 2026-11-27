"""End-to-end: ingestion → index → retrieval → generation → scoring → report (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from elevate_rag.cli.main import app
from elevate_rag.config import PipelineConfig, RouterConfig
from elevate_rag.data.prepare import DatasetStore
from elevate_rag.evaluation.harness import load_rows, run_questions
from elevate_rag.experiments.runner import ExperimentSpec, Variant, execute_run, expand
from elevate_rag.pipeline import ALL_MODES, ElevateRAG
from elevate_rag.reporting import report as R
from elevate_rag.routing.fallback import answer_routed
from elevate_rag.routing.router import Router
from elevate_rag.types import Query
from elevate_rag.utils import BudgetExceededError, ElevateError

pytestmark = pytest.mark.e2e


async def test_all_modes_answer_with_provenance_and_refusal(tmp_cfg: PipelineConfig) -> None:
    ds = DatasetStore.open(tmp_cfg.data_dir, "synthetic")
    s = ElevateRAG(tmp_cfg)
    await s.build_index(ds.documents(), dataset_version=ds.version)
    try:
        q = Query("q", "In which year was Velmora founded?")
        for mode in ALL_MODES:
            rec = await s.answer(q, mode)
            assert rec.error is None, (mode, rec.error)
            assert rec.context_tokens <= tmp_cfg.retrieval.context_budget
            assert all(c in s.store for c in rec.cited_chunk_ids)  # type: ignore[operator]
        v4 = await s.answer(q, "vector")
        assert "1712" in v4.answer and v4.cited_chunk_ids == ["syn-001::c0000"]
        assert v4.top_score is not None and v4.calls and v4.cost_usd == 0.0
        unans = await s.answer(
            Query("u", "What is the population of the moon colony Arkhaven?"), "v4"
        )
        assert unans.refused
        empty = await s.answer(Query("e", "?"), "v1")  # empty after tokenisation
        assert empty.error is None and empty.refused
        rules = Router(RouterConfig(level="rules"))
        routed = await answer_routed(
            s, Query("r", "Where was the founder of Lumenward born?"), rules
        )
        assert routed.route["level"] == "R1" and routed.mode == "local"
        fb = Router(RouterConfig(level="fallback", fallback_threshold=2.0))  # always escalates
        rec = await answer_routed(s, q, fb)
        assert rec.route["escalated"] and rec.route["initial_mode"] == "v4"
        assert rec.mode == "mix" and rec.route["final_mode"] == "mix"
        assert len(rec.calls) > len((await s.answer(q, "mix")).calls)  # both paths counted
    finally:
        await s.close()


async def test_empty_corpus_is_rejected(tmp_cfg: PipelineConfig) -> None:
    with pytest.raises(ElevateError, match="empty corpus"):
        await ElevateRAG(tmp_cfg).build_index([])


async def test_budget_cap_aborts_paid_generation(tmp_cfg: PipelineConfig) -> None:
    paid = tmp_cfg.models.model_copy(
        update={
            "generator": tmp_cfg.models.generator.model_copy(
                update={
                    "provider": "anthropic",
                    "model": "claude-sonnet-5-5",
                    "price_input_per_mtok": 2.0,
                    "price_output_per_mtok": 10.0,
                }
            )
        }
    )
    cfg = tmp_cfg.model_copy(update={"models": paid})
    ds = DatasetStore.open(cfg.data_dir, "synthetic")
    s = ElevateRAG(cfg)
    await s.build_index(ds.documents(), graph=False)
    with pytest.raises(BudgetExceededError):
        await s.answer(Query("q", "When was Velmora founded?"), "v4")
    await s.close()


async def test_run_questions_resumes_after_crash(tmp_cfg: PipelineConfig, tmp_path: Path) -> None:
    ds = DatasetStore.open(tmp_cfg.data_dir, "synthetic")
    s = ElevateRAG(tmp_cfg)
    await s.build_index(ds.documents(), graph=False)
    qs = ds.questions()
    out = tmp_path / "pred.jsonl"
    seen: list[str] = []

    async def answer(q: Query):  # type: ignore[no-untyped-def]
        seen.append(q.query_id)
        return await s.answer(q, "v1")

    await run_questions(answer, qs[:5], out, run_meta={"run_id": "r"})
    with out.open("a") as fh:
        fh.write('{"question_id": "truncated')  # simulated crash mid-write
    seen.clear()
    rows = await run_questions(answer, qs, out, run_meta={"run_id": "r"})
    assert len(seen) == len(qs) - 5 and len(rows) == len(qs)
    assert len(load_rows(out)) == len(qs)
    await s.close()


async def test_experiment_run_score_report_and_resume(
    tmp_cfg: PipelineConfig, tmp_path: Path
) -> None:
    (tmp_path / "pipe.yaml").write_text(yaml.safe_dump(tmp_cfg.model_dump(mode="json")))
    spec = ExperimentSpec(
        id="T",
        question="RQ1",
        pipeline="pipe.yaml",
        datasets=["synthetic"],
        variants=[
            Variant(name="V4", mode="v4"),
            Variant(name="local", mode="local"),
            Variant(name="R4", router={"level": "fallback"}, tune_on_dev=True),
        ],
    )
    runs_root = tmp_path / "runs"
    for r in expand(spec, tmp_path):
        summary = await execute_run(r, spec, runs_root)
        assert summary["overall"]["n"] == len(
            DatasetStore.open(tmp_cfg.data_dir, "synthetic").questions("test")
        )
        d = runs_root / "T" / r.run_id
        man = json.loads((d / "manifest.json").read_text())
        assert man["status"] == "complete" and man["demo"] is True
        assert man["definition"]["config_hash"] == r.cfg.config_hash()
        rows = [json.loads(x) for x in (d / "scored.jsonl").read_text().splitlines()]
        assert all({"run_id", "config_hash", "dataset_version", "scores"} <= set(x) for x in rows)
    tuning = json.loads(next(runs_root.rglob("fallback_tuning.json")).read_text())
    assert tuning["dev_questions"] > 0  # tuned on dev, never on test
    first = expand(spec, tmp_path)[0]
    before = (runs_root / "T" / first.run_id / "summary.json").stat().st_mtime
    await execute_run(first, spec, runs_root)  # complete runs are skipped
    assert (runs_root / "T" / first.run_id / "summary.json").stat().st_mtime == before

    loaded = R.load_runs(runs_root)
    table = R.results_table(loaded)
    comps = R.paired_comparisons(loaded, "V4")
    assert {c["variant"] for c in comps} == {"local", "R4"} and all("p_holm" in c for c in comps)
    fig = R.fig_accuracy_by_type(loaded, "synthetic", tmp_path / "rep" / "acc.png")
    assert fig is not None and fig.stat().st_size > 1000
    R.write_markdown_report(tmp_path / "rep" / "REPORT.md", table, comps, [fig])
    text = (tmp_path / "rep" / "REPORT.md").read_text()
    assert "DEMO OUTPUT" in text and "no clear difference" in text


def test_cli_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["data", "synthetic", "--data-dir", str(tmp_path / "data")])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["data", "info", "synthetic", "--data-dir", str(tmp_path / "data")])
    assert res.exit_code == 0 and '"synthetic": true' in res.output
    res = runner.invoke(app, ["data", "list"])
    assert res.exit_code == 0 and "hotpotqa" in res.output
    cfg = tmp_path / "p.yaml"
    from tests.conftest import offline_config

    c = offline_config(tmp_path)
    cfg.write_text(yaml.safe_dump(c.model_dump(mode="json")))
    res = runner.invoke(app, ["index", "build", str(cfg)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app, ["ask", str(cfg), "In which year was Velmora founded?", "--mode", "v4"]
    )
    assert res.exit_code == 0 and "1712" in res.output
    res = runner.invoke(
        app, ["ask", str(cfg), "Where was the founder of Velmora born?", "--router", "rules"]
    )
    assert res.exit_code == 0 and '"level": "R1"' in res.output
    res = runner.invoke(app, ["graph", str(cfg)])
    assert res.exit_code == 0 and "top_entities" in res.output
    res = runner.invoke(app, ["index", "validate", str(cfg)])
    assert res.exit_code == 0 and "gate_G3_coverage_ok" in res.output
    res = runner.invoke(
        app,
        [
            "router",
            "train",
            str(tmp_path / "r"),
            "--dataset",
            "synthetic",
            "--split",
            "test",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert res.exit_code != 0  # training on test is refused
    res = runner.invoke(app, ["env"])
    assert res.exit_code == 0 and "credentials_present" in res.output


def test_env_reports_a_hung_docker_daemon_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import subprocess

    def hung(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker info", timeout=20)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(subprocess, "run", hung)
    res = CliRunner().invoke(app, ["env"])
    assert res.exit_code == 0, res.output
    assert '"docker_daemon": false' in res.output
