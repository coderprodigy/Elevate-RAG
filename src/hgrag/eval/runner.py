"""Resumable, budget-aware execution of a pipeline over a dataset split.

Each run lives in ``runs/<experiment>/<dataset>__<pipeline>__<split>/``:

* ``config.json``: what ran (pipeline description, models, git SHA, timestamps)
* ``predictions.jsonl``: one :class:`Prediction` per line, appended as answers arrive
* ``usage.json``: API spend for this run, from the ledger

Re-running the same command resumes: questions already in ``predictions.jsonl`` are skipped.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hgrag.config import PROJECT_ROOT
from hgrag.core.hashing import stable_hash
from hgrag.data.schema import Query, Split
from hgrag.eval.batch import BatchOutcome, read_models, run_resumable
from hgrag.pipelines.base import Pipeline, Prediction

logger = logging.getLogger(__name__)


class SplitLockedError(RuntimeError):
    """Evaluating on the test split before the experiment config is frozen (CLAUDE.md rule 5)."""


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    dataset: str
    split: Split
    pipeline: str
    limit: int | None = None
    frozen: bool = False
    tag: str = ""  # non-reference arm, e.g. "embedder=bge-small"

    @property
    def run_name(self) -> str:
        pipeline = f"{self.pipeline}[{self.tag}]" if self.tag else self.pipeline
        return f"{self.dataset}__{pipeline}__{self.split}"

    @property
    def run_id(self) -> str:
        """Ledger run ID. Stable across resumes, so the per-run budget covers the whole run."""
        return f"{self.experiment}.{self.run_name}"


def check_split_allowed(spec: RunSpec) -> None:
    if spec.split == "test" and not spec.frozen:
        raise SplitLockedError(
            f"experiment '{spec.experiment}' is not frozen; tune on dev, then set `frozen: true` "
            f"in configs/experiments/{spec.experiment}.yaml at the gate review"
        )


def select_queries(queries: Sequence[Query], limit: int | None) -> list[Query]:
    """The first ``limit`` queries in a fixed pseudo-random order (mixes query types)."""
    ordered = sorted(queries, key=lambda q: stable_hash(q.qid))
    return ordered if limit is None else ordered[:limit]


@dataclass(frozen=True)
class RunPlan:
    n_selected: int
    n_done: int
    n_pending: int
    max_cost_usd: float  # upper bound; cached calls count as 0


RunOutcome = BatchOutcome  # answers written, failures, budget-stop reason


class EvalRunner:
    def __init__(self, pipeline: Pipeline, out_dir: Path, *, concurrency: int = 8) -> None:
        self._pipeline = pipeline
        self._out_dir = out_dir
        self._concurrency = concurrency

    @property
    def predictions_path(self) -> Path:
        return self._out_dir / "predictions.jsonl"

    def done_qids(self) -> set[str]:
        return {p.qid for p in load_predictions(self.predictions_path)}

    def plan(self, queries: Sequence[Query]) -> RunPlan:
        done = self.done_qids()
        pending = [q for q in queries if q.qid not in done]
        return RunPlan(
            n_selected=len(queries),
            n_done=len(queries) - len(pending),
            n_pending=len(pending),
            max_cost_usd=sum(self._pipeline.estimate_cost(q) for q in pending),
        )

    async def run(self, queries: Sequence[Query]) -> RunOutcome:
        done = self.done_qids()
        pending = [q for q in queries if q.qid not in done]
        try:
            return await run_resumable(
                pending,
                self._pipeline.answer,
                self.predictions_path,
                concurrency=self._concurrency,
                label=lambda q: q.qid,
            )
        finally:
            close = getattr(self._pipeline, "aclose", None)  # graph pipelines hold storages
            if close is not None:
                await close()


def load_predictions(path: Path) -> list[Prediction]:
    return read_models(path, Prediction)


def write_run_config(out_dir: Path, spec: RunSpec, pipeline: dict[str, Any]) -> None:
    """Record what ran. Keeps the first ``started_at`` across resumes."""
    path = out_dir / "config.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    now = datetime.now(UTC).isoformat()
    payload = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "pipeline": pipeline,
        "git_sha": _git_sha(),
        "started_at": previous.get("started_at", now),
        "updated_at": now,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "no-commits-yet"
