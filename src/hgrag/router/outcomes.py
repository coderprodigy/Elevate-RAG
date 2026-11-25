"""Per-question outcomes of every retrieval path, read from the per-mode runs (E2).

Routing is then evaluated offline: a router's accuracy and cost on a question are those of the
path it picked. This is exact (same cached answers) and costs nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hgrag.config import AppConfig
from hgrag.eval.runner import load_predictions


@dataclass(frozen=True)
class PathOutcome:
    correct: float  # primary metric (judge score, else exact match / refusal)
    tokens: int
    cost_usd: float
    top_score: float | None  # best retrieval score among chunks (reranker score for V4)
    refused: bool


@dataclass(frozen=True)
class OutcomeTable:
    """qid → path → outcome, for questions answered by *every* path."""

    dataset: str
    split: str
    outcomes: dict[str, dict[str, PathOutcome]]

    @property
    def qids(self) -> list[str]:
        return sorted(self.outcomes)

    def mean_tokens(self) -> dict[str, float]:
        paths = next(iter(self.outcomes.values()), {}).keys()
        n = max(len(self.outcomes), 1)
        return {p: sum(o[p].tokens for o in self.outcomes.values()) / n for p in paths}

    def cost_order(self) -> list[str]:
        """Paths from cheapest to most expensive (mean tokens per question)."""
        means = self.mean_tokens()
        return sorted(means, key=lambda p: means[p])


def run_dir(config: AppConfig, experiment: str, dataset: str, pipeline: str, split: str) -> Path:
    return config.paths.runs_dir / experiment / f"{dataset}__{pipeline}__{split}"


def load_outcomes(
    config: AppConfig, experiment: str, dataset: str, split: str, paths: Mapping[str, str]
) -> OutcomeTable:
    per_path: dict[str, dict[str, PathOutcome]] = {}
    for path, pipeline in paths.items():
        folder = run_dir(config, experiment, dataset, pipeline, split)
        scores_file = folder / "scores.jsonl"
        if not scores_file.is_file():
            raise FileNotFoundError(f"missing {scores_file}; run E2 for '{pipeline}' first")
        scores = {
            row["qid"]: row
            for row in (json.loads(line) for line in scores_file.read_text().splitlines() if line)
        }
        predictions = {p.qid: p for p in load_predictions(folder / "predictions.jsonl")}
        per_path[path] = {
            qid: PathOutcome(
                correct=float(row["correct"]),
                tokens=int(row["tokens"]),
                cost_usd=float(row["cost_usd"]),
                top_score=max(
                    (
                        c.score
                        for c in predictions[qid].contexts
                        if c.kind == "chunk" and c.score is not None
                    ),
                    default=None,
                )
                if qid in predictions
                else None,
                refused=bool(row["refused"]),
            )
            for qid, row in scores.items()
        }
    shared = set.intersection(*(set(o) for o in per_path.values())) if per_path else set()
    return OutcomeTable(
        dataset=dataset,
        split=split,
        outcomes={qid: {p: per_path[p][qid] for p in per_path} for qid in sorted(shared)},
    )
