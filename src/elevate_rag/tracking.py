"""Traceable results: a committed record per run, plus optional MLflow tracking (D-36).

After every completed run `record_run` writes

* ``<results_dir>/runs/<experiment>/<run_id>.json``: what was run (dataset version, variant,
  config hash, index key, every model with its pinned revision), from which code (git commit,
  branch, uncommitted changes), on what machine, and what came out (headline metrics with 95 %
  bootstrap CIs, judge agreement and validation status, efficiency, cost);
* ``<results_dir>/runs/INDEX.csv``: one line per run record, rebuilt from the records so it
  never drifts from them.

These are small text files meant to be committed. The raw per-question predictions stay in the
run directory (git-ignored); the record names it and the config hash, so any committed number
can be traced back to its exact inputs and re-run.

With MLflow enabled (`--mlflow` / ``ELEVATE_MLFLOW=1``, extra ``tracking``) each run is also
logged to an MLflow store (default ``sqlite:///<runs_root>/mlflow.db``) with the same
parameters, metrics and tags plus the run's summary and manifest as artifacts. Browse it with
``mlflow ui --backend-store-uri sqlite:///runs/mlflow.db``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from elevate_rag import __version__
from elevate_rag.utils import ConfigError, write_json

log = logging.getLogger(__name__)

# Headline metrics copied into the record (mean + CI + n) when the run has them.
RECORD_METRICS = (
    "correct",
    "correct_judge2",
    "em",
    "f1",
    "refusal_correct",
    "recall@10",
    "ndcg@10",
    "evidence_recall",
    "ragas_answer_correctness",
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
    "faithfulness",
    "context_recall",
    "context_precision",
)
INDEX_FIELDS = [
    "experiment",
    "run_id",
    "dataset",
    "variant",
    "split",
    "n",
    "em",
    "f1",
    "accuracy_judge",
    "accuracy_judge2",
    "judge_kappa",
    "judge_status",
    "recall@10",
    "ragas_faithfulness",
    "cost_usd",
    "git_commit",
    "git_dirty",
    "config_hash",
    "completed_at",
    "record",
]


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty flag of the checkout the code runs from (None if not a repo)."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _models(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for role, m in (config.get("models") or {}).items():
        if isinstance(m, dict):
            out[role] = {k: m.get(k) for k in ("provider", "model", "revision", "family")}
    return out


def build_record(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    run_dir: Path,
    git: dict[str, Any],
) -> dict[str, Any]:
    """`git` is the checkout writing the record; the run's own code version comes from its
    manifest when recorded there (runs from before that are flagged `git_from_run: false`)."""
    d = manifest["definition"]
    env = dict(manifest.get("environment") or {})
    run_git = env.pop("git", None)
    code_git = {**(run_git or git), "from_run": run_git is not None}
    overall = summary.get("overall", {})
    metrics = {k: overall[k] for k in RECORD_METRICS if isinstance(overall.get(k), dict)}
    ledger = summary.get("ledger") or {}
    return {
        "record_version": 1,
        "run_id": d["run_id"],
        "experiment": d["experiment"],
        "research_question": d.get("research_question"),
        "dataset": d["dataset"],
        "dataset_version": summary.get("dataset_version"),
        "variant": d["variant"],
        "split": d["split"],
        "limit": d.get("limit"),
        "n": overall.get("n"),
        "config_hash": d["config_hash"],
        "index_key": d["index_key"],
        "models": _models(d.get("config", {})),
        "metrics": metrics,
        "judge_agreement": summary.get("judge_agreement"),
        "judge_validation": summary.get("judge_validation"),
        "efficiency": summary.get("efficiency"),
        "error_rate": summary.get("error_rate"),
        "refusal_rate": summary.get("refusal_rate"),
        "cost_usd": ledger.get("spent_usd"),
        "code": {
            "elevate_rag": env.get("elevate_rag", __version__),
            **{f"git_{k}": v for k, v in code_git.items()},
        },
        # No host name or absolute paths: records are committed and must not identify machines.
        "environment": {**env, "slurm_job_id": os.getenv("SLURM_JOB_ID")},
        "created_at": manifest.get("created_at"),
        "completed_at": manifest.get("updated_at"),
        "demo": manifest.get("demo", False),
        "raw_run_dir": f"{d['experiment']}/{d['run_id']}",  # relative to the runs root
        "mlflow_run_id": None,
    }


def write_record(results_dir: Path, record: dict[str, Any]) -> Path:
    path = results_dir / "runs" / record["experiment"] / f"{record['run_id']}.json"
    write_json(path, record)
    rebuild_index(results_dir)
    return path


def _mean(rec: dict[str, Any], key: str) -> float | None:
    v = rec.get("metrics", {}).get(key)
    return v.get("mean") if isinstance(v, dict) else None


def rebuild_index(results_dir: Path) -> Path:
    root = results_dir / "runs"
    rows = []
    for p in sorted(root.glob("*/*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:  # pragma: no cover - a concurrent writer's partial file
            log.warning("skipping unreadable run record %s", p)
            continue
        jv = rec.get("judge_validation") or {}
        code = rec.get("code") or {}
        rows.append(
            {
                "experiment": rec["experiment"],
                "run_id": rec["run_id"],
                "dataset": rec["dataset"],
                "variant": (rec.get("variant") or {}).get("name"),
                "split": rec.get("split"),
                "n": rec.get("n"),
                "em": _mean(rec, "em"),
                "f1": _mean(rec, "f1"),
                "accuracy_judge": _mean(rec, "correct"),
                "accuracy_judge2": _mean(rec, "correct_judge2"),
                "judge_kappa": (rec.get("judge_agreement") or {}).get("kappa"),
                "judge_status": jv.get("status"),
                "recall@10": _mean(rec, "recall@10"),
                "ragas_faithfulness": _mean(rec, "ragas_faithfulness"),
                "cost_usd": rec.get("cost_usd"),
                "git_commit": (code.get("git_commit") or "")[:12],
                "git_dirty": code.get("git_dirty"),
                "config_hash": rec.get("config_hash"),
                "completed_at": rec.get("completed_at"),
                "record": str(p.relative_to(root)),
            }
        )
    out = root / "INDEX.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".INDEX.csv.{os.getpid()}.tmp")  # parallel shards: atomic replace
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    tmp.replace(out)
    return out


def default_tracking_uri(runs_root: Path) -> str:
    return os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(runs_root / 'mlflow.db').resolve()}"


def log_mlflow(record: dict[str, Any], run_dir: Path, tracking_uri: str) -> str:
    """Log one run record to MLflow; returns the MLflow run id. Artifacts go next to the store."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("MLflow tracking needs the extra: uv sync --extra tracking") from exc
    mlflow.set_tracking_uri(tracking_uri)
    name = f"elevate-rag/{record['experiment']}"
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        artifacts = None
        if tracking_uri.startswith("sqlite:///"):
            db = Path(tracking_uri[len("sqlite:///") :])
            artifacts = (db.parent / "mlflow-artifacts").resolve()
            artifacts.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(
            name, artifact_location=artifacts.as_uri() if artifacts else None
        )
    else:
        exp_id = exp.experiment_id
    code = record.get("code") or {}
    tags = {
        "run_id": record["run_id"],
        "config_hash": record["config_hash"],
        "index_key": record["index_key"],
        "dataset_version": str(record.get("dataset_version")),
        "git_commit": str(code.get("git_commit")),
        "git_dirty": str(code.get("git_dirty")),
        "judge_status": str((record.get("judge_validation") or {}).get("status")),
        "demo": str(record.get("demo")),
    }
    params: dict[str, Any] = {
        "dataset": record["dataset"],
        "variant": (record.get("variant") or {}).get("name"),
        "mode": (record.get("variant") or {}).get("mode"),
        "split": record["split"],
        "limit": record.get("limit"),
    }
    for role, m in (record.get("models") or {}).items():
        if m:
            params[f"model.{role}"] = f"{m['model']}@{m.get('revision') or '-'}"
    metrics: dict[str, float] = {}
    for k, v in (record.get("metrics") or {}).items():
        for part in ("mean", "ci_low", "ci_high"):
            if isinstance(v.get(part), int | float):
                metrics[f"{k.replace('@', '_at_')}.{part}"] = float(v[part])
    agreement = record.get("judge_agreement") or {}
    if isinstance(agreement.get("kappa"), int | float):
        metrics["judge_kappa"] = float(agreement["kappa"])
    for k, v in (record.get("efficiency") or {}).items():
        if isinstance(v, int | float) and not isinstance(v, bool):
            metrics[f"efficiency.{k}"] = float(v)
    if isinstance(record.get("cost_usd"), int | float):
        metrics["cost_usd"] = float(record["cost_usd"])
    with mlflow.start_run(experiment_id=exp_id, run_name=record["run_id"], tags=tags) as mr:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        for f in ("summary.json", "manifest.json", "ledger.json"):
            if (run_dir / f).exists():
                mlflow.log_artifact(str(run_dir / f))
        mlflow.log_dict(record, "record.json")
        return str(mr.info.run_id)


def record_run(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    run_dir: Path,
    results_dir: Path,
    *,
    mlflow_uri: str | None = None,
) -> Path:
    """Write the committed record for a completed run (and log it to MLflow if enabled)."""
    record = build_record(manifest, summary, run_dir, git_info())
    if mlflow_uri:
        try:
            record["mlflow_run_id"] = log_mlflow(record, run_dir, mlflow_uri)
        except ConfigError:
            raise
        except Exception as exc:  # tracking must never lose a finished run's record
            log.warning("MLflow logging failed for %s: %s", record["run_id"], exc)
    return write_record(results_dir, record)
