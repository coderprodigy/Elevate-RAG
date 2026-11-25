"""Ranked-retrieval metrics against gold IDs (binary relevance).

IDs may be chunk IDs or document IDs; callers decide the level. Duplicate IDs in a ranking
(e.g. several chunks of one document) count once, at their best rank.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def dedupe_ranking(ranked_ids: Sequence[str]) -> list[str]:
    """Keep the first (best-ranked) occurrence of each ID."""
    return list(dict.fromkeys(ranked_ids))


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Collection[str], k: int) -> float | None:
    """Share of gold IDs found in the top ``k``. None when there is no gold (undefined)."""
    if not gold_ids:
        return None
    top = set(dedupe_ranking(ranked_ids)[:k])
    return len(top & set(gold_ids)) / len(set(gold_ids))


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Collection[str], k: int) -> float | None:
    """1.0 if any gold ID is in the top ``k`` (DPR "top-k accuracy"). None when there is no gold."""
    if not gold_ids:
        return None
    return float(bool(set(dedupe_ranking(ranked_ids)[:k]) & set(gold_ids)))


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Collection[str], k: int = 10) -> float | None:
    """Normalised discounted cumulative gain with binary relevance. None when there is no gold."""
    gold = set(gold_ids)
    if not gold:
        return None
    ranking = dedupe_ranking(ranked_ids)[:k]
    dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(ranking) if i in gold)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / ideal
