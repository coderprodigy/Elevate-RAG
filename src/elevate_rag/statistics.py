"""Paired, question-level statistical summaries."""

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


def interval(values: Sequence[float], *, seed: int = 42, resamples: int = 1000) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not len(values):
        return {"n": 0, "estimate": None, "lower": None, "upper": None}
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or not np.isfinite(observations).all():
        raise ValueError("observations must be finite scalars")
    rng = np.random.default_rng(seed)
    averages = [
        float(rng.choice(observations, size=len(values), replace=True).mean())
        for _ in range(resamples)
    ]
    lower, upper = np.quantile(averages, [0.025, 0.975])
    return {
        "n": len(values),
        "estimate": float(observations.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def paired(left: dict[str, float], right: dict[str, float], *, seed: int = 42) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("paired comparison requires exactly aligned question IDs")
    return interval([right[key] - left[key] for key in sorted(left)], seed=seed)


def mcnemar(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("McNemar requires exactly aligned question IDs")
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    discordant = left_only + right_only
    if discordant:
        # Integer summation before the division avoids underflow during term evaluation.
        prob = min(
            1.0,
            2
            * (
                sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
                / (1 << discordant)
            ),
        )
    else:
        prob = 1.0
    return {"left_only": left_only, "right_only": right_only, "p_value": prob}


def holm(probabilities: dict[str, float]) -> dict[str, float]:
    if any(not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("p-values must be in [0, 1]")
    adjusted = {}
    previous = 0.0
    for index, (key, value) in enumerate(sorted(probabilities.items(), key=lambda x: (x[1], x[0]))):
        previous = min(1.0, max(previous, value * (len(probabilities) - index)))
        adjusted[key] = previous
    return adjusted


def minimum_detectable_effect(
    dev_left: dict[str, bool],
    dev_right: dict[str, bool],
    sample_size: int,
    *,
    power: float = 0.8,
) -> dict[str, float | int | None]:
    if sample_size < 1 or not 0.5 < power < 1:
        raise ValueError("power analysis requires a sample size and a power between 0.5 and 1")
    counts = mcnemar(dev_left, dev_right)
    if not dev_left or counts["left_only"] + counts["right_only"] == 0:
        return {
            "development_n": len(dev_left),
            "target_n": sample_size,
            "discordance": None,
            "approximate_mde": None,
        }
    rate = (counts["left_only"] + counts["right_only"]) / len(dev_left)
    normal = NormalDist()
    effect = (normal.inv_cdf(0.975) + normal.inv_cdf(power)) * math.sqrt(rate / sample_size)
    return {
        "development_n": len(dev_left),
        "target_n": sample_size,
        "discordance": rate,
        "approximate_mde": effect,
    }
