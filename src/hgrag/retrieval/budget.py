"""Context budget (decision D4): every path is trimmed to the same token cap before generation."""

from __future__ import annotations

from collections.abc import Sequence


def apply_token_budget[T](items: Sequence[T], tokens: Sequence[int], budget: int) -> list[T]:
    """Keep items in rank order while they fit in ``budget``; always keep the first one."""
    kept: list[T] = []
    used = 0
    for item, n in zip(items, tokens, strict=True):
        if kept and used + n > budget:
            break
        kept.append(item)
        used += n
    return kept
