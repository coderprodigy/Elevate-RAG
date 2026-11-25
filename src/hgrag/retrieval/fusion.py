"""Reciprocal rank fusion (Cormack et al., 2009): score = Σ 1 / (k + rank), rank from 1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, limit: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one (id, fused score) list, best first; ties keep first-seen."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: -pair[1])  # stable sort keeps insertion order
    return fused if limit is None else fused[:limit]
