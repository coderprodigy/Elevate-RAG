"""Lenient JSON extraction from model replies (models sometimes wrap JSON in prose or fences)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JSONExtractionError(ValueError):
    """The reply contains no parseable JSON object or array."""


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (inside a code fence if there is one)."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise JSONExtractionError(f"no JSON found in reply: {text[:120]!r}")
