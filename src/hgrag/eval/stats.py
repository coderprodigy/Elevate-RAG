"""Statistics for paired system comparisons (plan "Statistics" section). Standard library only.

* :func:`bootstrap_ci`: percentile CI of a mean, resampling questions.
* :func:`mcnemar`: paired test for correct/wrong outcomes (exact for few discordant pairs).
* :func:`paired_bootstrap`: CI and p-value for the mean difference of continuous scores.
* :func:`holm`: Holm-Bonferroni adjusted p-values.
* :func:`min_detectable_effect`: smallest accuracy difference a paired design can detect.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean

_EXACT_LIMIT = 25  # below this many discordant pairs, McNemar uses the exact binomial test


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 750
) -> Interval:
    if not values:
        return Interval(math.nan, math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    return Interval(
        estimate=fmean(values),
        low=means[int(alpha / 2 * n_resamples)],
        high=means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
    )


@dataclass(frozen=True)
class McNemarResult:
    a_only: int  # questions only system A got right
    b_only: int
    p_value: float
    exact: bool


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcomes must have equal length")
    a_only = sum(a and not b for a, b in zip(a_correct, b_correct, strict=True))
    b_only = sum(b and not a for a, b in zip(a_correct, b_correct, strict=True))
    discordant = a_only + b_only
    if discordant == 0:
        return McNemarResult(a_only, b_only, 1.0, exact=True)
    if discordant < _EXACT_LIMIT:
        tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
        return McNemarResult(a_only, b_only, min(1.0, 2 * tail / 2**discordant), exact=True)
    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant  # continuity-corrected, 1 df
    return McNemarResult(a_only, b_only, math.erfc(math.sqrt(statistic / 2)), exact=False)


@dataclass(frozen=True)
class PairedDifference:
    mean_diff: float  # mean(a - b)
    low: float
    high: float
    p_value: float  # two-sided, from the bootstrap distribution


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 750,
) -> PairedDifference:
    if len(a) != len(b) or not a:
        raise ValueError("paired scores must be non-empty and of equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(rng.choices(diffs, k=n)) for _ in range(n_resamples))
    observed = fmean(diffs)
    # Two-sided p: how often the resampled mean falls on the other side of zero.
    opposite = sum(m <= 0 for m in means) if observed > 0 else sum(m >= 0 for m in means)
    return PairedDifference(
        mean_diff=observed,
        low=means[int(alpha / 2 * n_resamples)],
        high=means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
        p_value=min(1.0, 2 * opposite / n_resamples) if observed != 0 else 1.0,
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def min_detectable_effect(
    n: int, discordant_rate: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Approximate smallest accuracy difference detectable by McNemar with ``n`` paired questions.

    ``discordant_rate`` is the share of questions where the two systems disagree (from dev runs).
    """
    z = NormalDist().inv_cdf
    return (z(1 - alpha / 2) + z(power)) * math.sqrt(discordant_rate / n)
