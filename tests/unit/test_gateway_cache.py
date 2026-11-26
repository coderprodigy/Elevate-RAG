"""Unit tests for the content-hash cache (Req 4.1, 4.2)."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from elevate.gateway.cache import (
    CACHE_DIR_ENV,
    CacheKey,
    CacheNamespace,
    SQLiteCache,
    default_cache_path,
)


class Counter:
    def __init__(self, value: object) -> None:
        self.calls = 0
        self.value = value

    def __call__(self) -> object:
        self.calls += 1
        return self.value


def key(content: object = "prompt", model: str = "m1", **params: object) -> CacheKey:
    return CacheKey.for_request(CacheNamespace.GENERATION, content, model, params)


@pytest.fixture
def cache(tmp_path: Path) -> SQLiteCache:
    return SQLiteCache(tmp_path / "c.sqlite3")


def test_second_call_hits_and_skips_thunk(cache: SQLiteCache) -> None:
    thunk = Counter({"text": "answer", "tokens": [1, 2]})
    first = cache.get_or_call_result(key(temperature=0), thunk)
    second = cache.get_or_call_result(key(temperature=0), thunk)
    assert thunk.calls == 1
    assert (first.hit, second.hit) == (False, True)
    assert first.value == second.value == {"text": "answer", "tokens": [1, 2]}
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)
    assert cache.stats.by_namespace["generation"] == {"hits": 1, "misses": 1}


def test_miss_returns_json_round_tripped_value(cache: SQLiteCache) -> None:
    # Tuples become lists on both miss and hit, so the two results are identical.
    first = cache.get_or_call(key(), lambda: (1, 2))
    assert first == [1, 2]
    assert cache.get_or_call(key(), lambda: (9, 9)) == first


@pytest.mark.parametrize(
    "other",
    [
        key(content="other prompt"),
        key(model="m2"),
        key(temperature=0.7),
        CacheKey.for_request(CacheNamespace.JUDGE, "prompt", "m1", {}),
    ],
)
def test_any_key_component_change_is_a_miss(cache: SQLiteCache, other: CacheKey) -> None:
    cache.get_or_call(key(), lambda: "a")
    assert cache.get_or_call_result(other, lambda: "b").value == "b"
    assert len(cache) == 2


def test_digest_ignores_param_order() -> None:
    a = CacheKey.for_request("embedding", ["x"], "e", {"a": 1, "b": 2})
    b = CacheKey.for_request("embedding", ["x"], "e", {"b": 2, "a": 1})
    assert a.digest == b.digest


def test_unknown_namespace_rejected() -> None:
    with pytest.raises(ValueError):
        CacheKey.for_request("nope", "x", "m")


def test_thunk_error_is_not_cached(cache: SQLiteCache) -> None:
    def boom() -> object:
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        cache.get_or_call(key(), boom)
    assert key() not in cache
    assert cache.get_or_call(key(), lambda: "ok") == "ok"


def test_non_json_value_rejected_and_not_stored(cache: SQLiteCache) -> None:
    with pytest.raises(TypeError):
        cache.get_or_call(key(), lambda: object())
    with pytest.raises(ValueError):
        cache.get_or_call(key(content="nan"), lambda: float("nan"))
    assert len(cache) == 0


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite3"
    SQLiteCache(path).get_or_call(key(), lambda: "stored")
    thunk = Counter("fresh")
    assert SQLiteCache(path).get_or_call(key(), thunk) == "stored"
    assert thunk.calls == 0


def test_uses_wal_mode(cache: SQLiteCache) -> None:
    import sqlite3

    with sqlite3.connect(cache.path) as conn:
        (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
    assert mode == "wal"


def test_default_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "envcache"))
    assert default_cache_path() == tmp_path / "envcache" / "cache.sqlite3"
    assert SQLiteCache().path.parent.is_dir()
    monkeypatch.delenv(CACHE_DIR_ENV)
    assert default_cache_path() == Path(".cache") / "cache.sqlite3"


def test_concurrent_threads_agree(cache: SQLiteCache) -> None:
    def work(i: int) -> object:
        return cache.get_or_call(key(content=i % 5), lambda: {"i": i % 5})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(80)))
    assert results == [{"i": i % 5} for i in range(80)]
    assert len(cache) == 5


def _process_worker(path: str, n: int) -> None:
    c = SQLiteCache(path)
    for i in range(n):
        c.get_or_call(key(content=i), lambda i=i: i)


def test_concurrent_processes_share_file(tmp_path: Path) -> None:
    path = str(tmp_path / "shared.sqlite3")
    SQLiteCache(path)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_process_worker, args=(path, 20)) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert len(SQLiteCache(path)) == 20
