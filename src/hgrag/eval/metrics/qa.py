"""Exact match and token F1 with SQuAD-style answer normalisation.

Both take the best score over all acceptable reference answers.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD v1.1)."""
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))


def token_f1(prediction: str, references: Sequence[str]) -> float:
    return max((_f1(prediction, ref) for ref in references), default=0.0)


def _f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
