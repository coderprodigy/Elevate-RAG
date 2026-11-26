from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elevate.core import tokens
from elevate.core.errors import ConfigError
from elevate.core.ids import canonical_json, chunk_id, content_hash, make_id
from elevate.core.tokens import FallbackTokenCounter, TokenCounter, get_token_counter

# ---------------------------------------------------------------- ids


def test_ids_are_deterministic_and_prefixed():
    a = chunk_id("doc-1", 0, 10, "hello")
    assert a == chunk_id("doc-1", 0, 10, "hello")
    assert a.startswith("chunk-") and len(a) == len("chunk-") + 16


def test_ids_change_with_any_part():
    base = chunk_id("doc-1", 0, 10, "hello")
    assert base != chunk_id("doc-2", 0, 10, "hello")
    assert base != chunk_id("doc-1", 1, 10, "hello")
    assert base != chunk_id("doc-1", 0, 10, "hellO")


def test_parts_do_not_collide_by_concatenation():
    assert make_id("x", "ab", "c") != make_id("x", "a", "bc")


def test_content_hash_ignores_dict_key_order():
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_make_id_validates_arguments():
    with pytest.raises(ValueError):
        make_id("", "x")
    with pytest.raises(ValueError):
        make_id("x", "y", length=4)


# ---------------------------------------------------------------- tokens


def test_fallback_counter_basic_counts():
    c = FallbackTokenCounter()
    assert c.count("") == 0
    assert c.count("Hello, world!") == 6  # Hell o , worl d ! -> letter runs cut at 4
    assert c.count("  \n\t ") == 0


@given(st.text(max_size=300), st.integers(min_value=0, max_value=80))
def test_fallback_truncate_is_prefix_within_budget(text: str, k: int):
    c = FallbackTokenCounter()
    out = c.truncate(text, k)
    assert text.startswith(out)
    assert c.count(out) <= k
    if c.count(text) <= k:
        assert out == text


def test_default_counter_is_cached_and_usable():
    counter = get_token_counter()
    assert isinstance(counter, TokenCounter)
    assert counter is get_token_counter()
    assert counter.count("hello world") > 0
    assert counter.count(counter.truncate("one two three four five", 2)) <= 2


def test_fallback_mode_forced_by_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(tokens.TOKENIZER_ENV, "fallback")
    assert get_token_counter().name == FallbackTokenCounter.name


def test_offline_without_cache_uses_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(tokens.TOKENIZER_ENV, "auto")
    monkeypatch.setenv("ELEVATE_OFFLINE", "1")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))  # empty cache
    tokens._load_counter.cache_clear()
    try:
        assert get_token_counter().name == FallbackTokenCounter.name
    finally:
        tokens._load_counter.cache_clear()


def test_invalid_mode_raises_config_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(tokens.TOKENIZER_ENV, "bogus")
    with pytest.raises(ConfigError, match=tokens.TOKENIZER_ENV):
        get_token_counter()


def test_tiktoken_truncate_when_available():
    if not tokens._tiktoken_cached(tokens.DEFAULT_ENCODING):
        pytest.skip("tiktoken encoding not cached locally")
    counter = tokens.TiktokenCounter()
    text = "Elevate-RAG builds one shared index. Ünïcödé 🙂 text too."
    for k in range(counter.count(text) + 1):
        out = counter.truncate(text, k)
        assert text.startswith(out) and counter.count(out) <= k
    assert counter.truncate(text, 10_000) == text
