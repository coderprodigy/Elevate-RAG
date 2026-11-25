"""Token → USD arithmetic."""

from __future__ import annotations

from hgrag.config import Price

_PER_MILLION = 1_000_000
# Conservative chars-per-token ratio for English, used only for pre-call budget estimates.
_CHARS_PER_TOKEN_ESTIMATE = 3.0


def cost_usd(
    price: Price, input_tokens: int, output_tokens: int, cached_input_tokens: int
) -> float:
    """Actual cost of one call. Cached input bills at the cached rate when one is defined."""
    cached_rate = (
        price.cached_input_per_mtok
        if price.cached_input_per_mtok is not None
        else price.input_per_mtok
    )
    return (
        input_tokens * price.input_per_mtok
        + cached_input_tokens * cached_rate
        + output_tokens * price.output_per_mtok
    ) / _PER_MILLION


def estimate_max_cost_usd(price: Price, prompt_chars: int, max_tokens: int) -> float:
    """Upper-bound estimate before a call: prompt from character count, output at ``max_tokens``."""
    est_input_tokens = int(prompt_chars / _CHARS_PER_TOKEN_ESTIMATE) + 1
    return cost_usd(price, est_input_tokens, max_tokens, 0)
