"""Latency, token and cost summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100] (same as numpy's default)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def usd_per_1k_queries(costs: Sequence[float]) -> float:
    return 1000 * sum(costs) / len(costs) if costs else math.nan
