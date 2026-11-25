"""Router metrics (E3 / RQ3): routing accuracy, macro-F1, regret, tokens saved."""

from __future__ import annotations

from collections.abc import Sequence


def routing_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    _check(predicted, gold)
    return sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold)


def macro_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Unweighted mean F1 over every label seen in ``gold`` or ``predicted``."""
    _check(predicted, gold)
    scores = []
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(p == g == label for p, g in zip(predicted, gold, strict=True))
        fp = sum(p == label != g for p, g in zip(predicted, gold, strict=True))
        fn = sum(g == label != p for p, g in zip(predicted, gold, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def regret(oracle_accuracy: float, router_accuracy: float) -> float:
    """How much accuracy the router leaves on the table versus perfect routing."""
    return oracle_accuracy - router_accuracy


def tokens_saved(router_tokens: float, baseline_tokens: float) -> float:
    """Fraction of tokens saved versus always running the baseline (e.g. ``mix``)."""
    return 1 - router_tokens / baseline_tokens if baseline_tokens else 0.0


def _check(predicted: Sequence[str], gold: Sequence[str]) -> None:
    if len(predicted) != len(gold) or not gold:
        raise ValueError("predicted and gold labels must be non-empty and of equal length")
