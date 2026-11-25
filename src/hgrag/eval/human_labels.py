"""Human labels for judge validation (protocol rule 5) and Cohen's kappa.

Both labellers see the *same* blind sample (same seed): question, references and answer, with no
system name. Labels go to ``runs/<experiment>/labels/<labeller>.jsonl``.
Judge validation passes when kappa(human, judge) ≥ 0.6 for every labeller.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from hgrag.core.hashing import stable_hash
from hgrag.data.schema import Query
from hgrag.pipelines.base import Prediction

KAPPA_THRESHOLD = 0.6


class LabelItem(BaseModel):
    item_id: str
    run: str
    qid: str
    question: str
    references: list[str]
    answer: str


class HumanLabel(BaseModel):
    item_id: str
    run: str
    qid: str
    labeller: str
    verdict: str  # correct | partial | wrong


def build_label_sample(
    runs: Sequence[tuple[str, dict[str, Query], Sequence[Prediction]]], n: int, seed: int
) -> list[LabelItem]:
    """A deterministic blind sample of ``n`` answered (non-refused, answerable) items."""
    items = [
        LabelItem(
            item_id=stable_hash([run_name, p.qid])[:12],
            run=run_name,
            qid=p.qid,
            question=queries[p.qid].question,
            references=queries[p.qid].answers,
            answer=p.answer,
        )
        for run_name, queries, predictions in runs
        for p in predictions
        if p.qid in queries and queries[p.qid].answerable and not p.refused
    ]
    items.sort(key=lambda item: item.item_id)
    random.Random(seed).shuffle(items)
    return items[:n]


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Agreement between two raters beyond chance. 1 = perfect, 0 = chance level."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    expected = sum(counts_a[k] * counts_b[k] for k in counts_a.keys() | counts_b.keys()) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def labels_path(experiment_dir: Path, labeller: str) -> Path:
    return experiment_dir / "labels" / f"{labeller}.jsonl"
