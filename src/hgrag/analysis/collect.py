"""Load every run of an experiment into flat records (one per run, plus per-question scores)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hgrag.config import AppConfig
from hgrag.eval.scoring import QuestionScore, RunMetrics


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    dataset: str
    pipeline: str  # including any arm tag, e.g. "v4[embedder=bge-small]"
    split: str
    metrics: RunMetrics
    scores: list[QuestionScore]
    config: dict[str, Any]

    @property
    def by_qid(self) -> dict[str, QuestionScore]:
        return {s.qid: s for s in self.scores}


def load_experiment_runs(
    config: AppConfig, experiment: str, split: str | None = None
) -> list[RunRecord]:
    root = config.paths.runs_dir / experiment
    records = []
    for folder in (
        sorted(p for p in root.glob("*") if (p / "metrics.json").is_file()) if root.is_dir() else []
    ):
        records.append(_load(folder))
    return [r for r in records if split is None or r.split == split]


def _load(folder: Path) -> RunRecord:
    dataset, pipeline, split = folder.name.split("__")
    scores_file = folder / "scores.jsonl"
    scores = (
        [
            QuestionScore.model_validate_json(line)
            for line in scores_file.read_text().splitlines()
            if line
        ]
        if scores_file.is_file()
        else []
    )
    return RunRecord(
        run_name=folder.name,
        dataset=dataset,
        pipeline=pipeline,
        split=split,
        metrics=RunMetrics.model_validate_json((folder / "metrics.json").read_text()),
        scores=scores,
        config=json.loads((folder / "config.json").read_text())
        if (folder / "config.json").is_file()
        else {},
    )
