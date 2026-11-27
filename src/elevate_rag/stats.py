"""Statistics required by the plan (FR-E10): bootstrap CIs, paired tests, multiple-comparison
correction, effect sizes, agreement and minimum detectable effect.

All functions take *question-aligned* arrays: element i of every system refers to the same
question. Alignment is checked, never assumed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float
    high: float
    n: int

    def fmt(self, pct: bool = True) -> str:
        f = 100.0 if pct else 1.0
        return f"{self.estimate * f:.1f} [{self.low * f:.1f}, {self.high * f:.1f}]"


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 13
) -> CI:
    """Percentile bootstrap over questions (plan: 1,000 resamples, 95 %)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("bootstrap_ci on empty sample")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(x.mean()), float(lo), float(hi), int(x.size))


@dataclass(frozen=True)
class PairedResult:
    diff: float  # mean(a) - mean(b)
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    n: int
    effect_size: float  # Cohen's h for proportions / d_z for continuous


def _check_aligned(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"paired samples must be aligned 1-D arrays, got {x.shape} vs {y.shape}")
    return x, y


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 1000, seed: int = 13
) -> PairedResult:
    """Paired bootstrap for continuous scores (F1, faithfulness). Two-sided p-value from the
    bootstrap distribution of the mean difference."""
    x, y = _check_aligned(a, b)
    d = x - y
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    centred = boots - boots.mean()
    p = float((np.abs(centred) >= abs(d.mean())).mean())
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return PairedResult(
        float(d.mean()),
        float(lo),
        float(hi),
        max(p, 1.0 / n_resamples),
        "paired_bootstrap",
        int(d.size),
        float(d.mean() / sd) if sd else 0.0,
    )


def mcnemar(a: Sequence[bool | int], b: Sequence[bool | int]) -> PairedResult:
    """McNemar's test for paired binary outcomes; exact binomial when discordant pairs < 25."""
    x, y = _check_aligned(a, b)
    x, y = x.astype(bool), y.astype(bool)
    n01 = int((x & ~y).sum())  # a right, b wrong
    n10 = int((~x & y).sum())
    disc = n01 + n10
    if disc == 0:
        p = 1.0
        test = "mcnemar_exact"
    elif disc < 25:
        p = float(sps.binomtest(n01, disc, 0.5).pvalue)
        test = "mcnemar_exact"
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / disc
        p = float(sps.chi2.sf(chi2, 1))
        test = "mcnemar_chi2_cc"
    pa, pb = x.mean(), y.mean()
    # Wald CI for the paired difference of proportions
    n = x.size
    diff = (n01 - n10) / n
    se = math.sqrt(max(disc / n**2 - (n01 - n10) ** 2 / n**3, 0.0))
    h = 2 * math.asin(math.sqrt(pa)) - 2 * math.asin(math.sqrt(pb))
    return PairedResult(float(diff), diff - 1.96 * se, diff + 1.96 * se, p, test, int(n), float(h))


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Holm step-down: returns (adjusted p, reject) in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return [(adjusted[i], adjusted[i] <= alpha) for i in range(m)]


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equally long, non-empty label lists")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mde_paired_proportions(
    n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Minimum detectable accuracy difference for a paired (McNemar) design.

    Uses the normal approximation: δ = (z_{1-α/2} + z_{power}) · sqrt(p_disc / n), where p_disc is
    the share of discordant pairs (estimate it on the dev split)."""
    if n <= 0 or not 0 < discordant_rate <= 1:
        raise ValueError("need n > 0 and 0 < discordant_rate <= 1")
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(z * math.sqrt(discordant_rate / n))


def power_check(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_test: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, float]:
    """Dev-set power check: discordance between two systems on dev → MDE at the test size."""
    x, y = _check_aligned([float(v) for v in correct_a], [float(v) for v in correct_b])
    disc = float((x != y).mean()) or 1.0 / max(len(x), 1)
    return {
        "dev_discordant_rate": disc,
        "n_test": n_test,
        "mde": mde_paired_proportions(n_test, disc, alpha, power),
    }
