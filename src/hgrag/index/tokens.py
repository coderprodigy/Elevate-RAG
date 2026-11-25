"""One fixed token unit (tiktoken ``cl100k_base``) for chunk sizes and context budgets.

Billing uses each provider's own tokenizer (from API usage fields); this unit only has to be the
same for every pipeline so that "512-token chunks" and "8k-token budget" mean the same thing.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Hard-split ``text`` into pieces of at most ``max_tokens`` (for over-long sentences)."""
    tokens = _encoding().encode(text, disallowed_special=())
    return [
        _encoding().decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]
