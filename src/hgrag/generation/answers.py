"""Answer conventions shared by all pipelines: the refusal phrase and answer cleanup."""

from __future__ import annotations

import re

REFUSAL = "insufficient information"

_REFUSAL_PATTERN = re.compile(r"\binsufficient\s+information\b", re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^\s*(final\s+)?answer\s*:\s*", re.IGNORECASE)
_CITATIONS = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def is_refusal(answer: str) -> bool:
    """True if the model gave the agreed refusal (anywhere in a short answer)."""
    return bool(_REFUSAL_PATTERN.search(answer)) and len(answer) < 200


def clean_answer(raw: str) -> str:
    """Strip a leading "Answer:" label and surrounding whitespace/quotes."""
    text = _ANSWER_PREFIX.sub("", raw.strip())
    return text.strip().strip('"').strip()


def strip_citations(answer: str) -> str:
    """Remove ``[1]``-style citation markers (for EM/F1 scoring)."""
    return _CITATIONS.sub("", answer).strip()
