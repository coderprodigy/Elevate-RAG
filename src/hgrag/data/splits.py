"""Deterministic, stratified dev/test splits that are frozen once written."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hgrag.core.hashing import stable_hash
from hgrag.data.schema import Query, Split


class FrozenSplitError(RuntimeError):
    """The frozen split file no longer matches the questions being built."""


def assign_splits(queries: Sequence[Query], dev_fraction: float, seed: int) -> dict[str, Split]:
    """Put ``dev_fraction`` of each query type in dev (rounded), the rest in test."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for query in queries:
        by_type[query.query_type].append(query.qid)

    rng = random.Random(seed)
    assignment: dict[str, Split] = {}
    for query_type in sorted(by_type):
        qids = sorted(by_type[query_type])
        rng.shuffle(qids)
        n_dev = round(len(qids) * dev_fraction)
        assignment.update({qid: "dev" for qid in qids[:n_dev]})
        assignment.update({qid: "test" for qid in qids[n_dev:]})
    return assignment


def load_or_create_splits(
    path: Path, queries: Sequence[Query], dev_fraction: float, seed: int
) -> dict[str, Split]:
    """Reuse the frozen ``splits.json`` if present; otherwise create and write it.

    A frozen file must cover exactly the current questions. Delete it on purpose to re-split.
    """
    qids = {q.qid for q in queries}
    if path.is_file():
        frozen: dict[str, Split] = json.loads(path.read_text(encoding="utf-8"))["splits"]
        if set(frozen) != qids:
            added, removed = len(qids - set(frozen)), len(set(frozen) - qids)
            raise FrozenSplitError(
                f"{path} is frozen but the questions changed (+{added} / -{removed}). "
                "Revert the sampling change, or delete the file deliberately to re-split."
            )
        return frozen

    splits = assign_splits(queries, dev_fraction, seed)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "sha256": stable_hash(splits),
        "splits": dict(sorted(splits.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return splits
