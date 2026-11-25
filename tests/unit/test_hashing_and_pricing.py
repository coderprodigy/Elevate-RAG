import pytest

from hgrag.config import Price
from hgrag.core.hashing import stable_hash
from hgrag.llm.pricing import cost_usd, estimate_max_cost_usd


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})


def test_stable_hash_changes_with_content() -> None:
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_cost_uses_cached_rate_when_defined() -> None:
    price = Price(input_per_mtok=2.0, output_per_mtok=10.0, cached_input_per_mtok=0.1)
    # 1M input at $2 + 1M cached at $0.1 + 0.5M output at $10 = 2 + 0.1 + 5
    assert cost_usd(price, 1_000_000, 500_000, 1_000_000) == pytest.approx(7.1)


def test_cached_tokens_bill_as_input_without_cached_rate() -> None:
    price = Price(input_per_mtok=0.1, output_per_mtok=0.5)
    assert cost_usd(price, 0, 0, 1_000_000) == pytest.approx(0.1)


def test_estimate_is_an_upper_bound_on_output() -> None:
    price = Price(input_per_mtok=0.0, output_per_mtok=1.0)
    assert estimate_max_cost_usd(price, prompt_chars=0, max_tokens=1_000_000) == pytest.approx(1.0)
