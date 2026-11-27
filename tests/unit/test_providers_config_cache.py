from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from elevate_rag.cache import KVCache
from elevate_rag.config import (
    ChunkingConfig,
    PipelineConfig,
    ProviderConfig,
    apply_overrides,
    load_pipeline_config,
)
from elevate_rag.providers.base import (
    RetryableError,
    UsageLedger,
    collect_calls,
    price,
    retry_async,
)
from elevate_rag.providers.embeddings import BaseEmbedder, HashEmbedder, make_embedder
from elevate_rag.providers.llm import (
    AnthropicLLM,
    OfflineLLM,
    OpenAICompatLLM,
    heuristic_extract,
    make_llm,
)
from elevate_rag.providers.rerank import LexicalReranker, make_reranker
from elevate_rag.utils import BudgetExceededError, ConfigError, ProviderError

# ------------------------------------------------------------------------------- config


def test_extends_overrides_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\ndataset: synthetic\nmodels:\n  embedder: {provider: hash, model: h}\n"
        "  generator: {provider: offline, model: g}\nretrieval: {final_k: 8}\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nname: child\nretrieval: {context_budget: 4096}\n"
    )
    cfg = load_pipeline_config(tmp_path / "child.yaml", {"retrieval.final_k": 5})
    assert cfg.name == "child" and cfg.retrieval.final_k == 5
    assert cfg.retrieval.context_budget == 4096
    assert (
        cfg.config_hash()
        == load_pipeline_config(tmp_path / "child.yaml", {"retrieval.final_k": 5}).config_hash()
    )
    other = load_pipeline_config(
        tmp_path / "child.yaml", {"models.embedder.model": "h2", "retrieval.final_k": 5}
    )
    assert other.index_key() != cfg.index_key()  # new embedder ⇒ new index (FR-M5)
    budget_only = load_pipeline_config(tmp_path / "child.yaml", {"retrieval.final_k": 6})
    assert budget_only.index_key() == cfg.index_key()  # query-time knobs keep the index


def test_invalid_configs_fail_loudly(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("name: x\nmodels: {}\nunknown_field: 1\n")
    with pytest.raises(ConfigError):
        load_pipeline_config(tmp_path / "bad.yaml")
    with pytest.raises(ConfigError):
        load_pipeline_config(tmp_path / "missing.yaml")
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_pipeline_config(tmp_path / "a.yaml")
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, {"a.b": 2})
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_tokens=10)


def test_repository_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "pipelines"
    files = sorted(root.glob("*.yaml"))
    assert files
    for f in files:
        load_pipeline_config(f)


# ------------------------------------------------------------------------------- cache


def test_cache_roundtrip_and_corruption_recovery(tmp_path: Path) -> None:
    c = KVCache(tmp_path, "ns")
    k = KVCache.key({"a": 1})
    c.set(k, {"v": [1, 2]})
    assert c.get(k) == {"v": [1, 2]} and len(c) == 1
    conn = sqlite3.connect(tmp_path / "ns.sqlite")
    conn.execute("UPDATE kv SET v = ? WHERE k = ?", ('{"v": [9', k))
    conn.commit()
    conn.close()
    c2 = KVCache(tmp_path, "ns")
    assert c2.get(k) is None and len(c2) == 0  # corrupt row detected and removed
    off = KVCache(None, "x")
    off.set("k", 1)
    assert off.get("k") is None and len(off) == 0


def test_cache_key_depends_on_every_field() -> None:
    assert KVCache.key({"model": "a", "t": "x"}) != KVCache.key({"model": "b", "t": "x"})
    assert KVCache.key({"b": 1, "a": 2}) == KVCache.key({"a": 2, "b": 1})


# ------------------------------------------------------------------------------- embedders


def _emb(**kw: Any) -> ProviderConfig:
    return ProviderConfig(provider="hash", model="h", **kw)


def test_hash_embedder_normalised_deterministic_and_cached(tmp_path: Path) -> None:
    ledger = UsageLedger()
    e = HashEmbedder(_emb(dimension=128), ledger=ledger, cache=KVCache(tmp_path, "e"))
    v = e.embed(["amber harbour", "software company"], kind="document")
    assert v.shape == (2, 128) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    v2 = e.embed(["amber harbour", "software company"], kind="document")
    assert np.allclose(v, v2, atol=1e-6)
    assert any(r.cached for r in ledger.records())
    sim = float(e.embed(["amber harbour"])[0] @ e.embed(["the amber harbour"])[0])
    assert sim > float(e.embed(["amber harbour"])[0] @ e.embed(["tidal turbine"])[0])


def test_embedding_dimension_mismatch_is_rejected() -> None:
    class Wrong(BaseEmbedder):
        def _embed_batch(self, texts: list[str], kind: Any) -> np.ndarray:
            return np.ones((len(texts), 7))

    with pytest.raises(ProviderError, match="dimension"):
        Wrong(_emb(dimension=8), ledger=UsageLedger()).embed(["x"])
    with pytest.raises(ConfigError):
        make_embedder(ProviderConfig(provider="nope", model="m"), ledger=UsageLedger())


# ------------------------------------------------------------------------------- ledger/budget


def test_budget_cap_blocks_paid_calls_before_spending() -> None:
    paid = ProviderConfig(
        provider="anthropic",
        model="claude-haiku-5-5",
        price_input_per_mtok=0.10,
        price_output_per_mtok=0.50,
    )
    ledger = UsageLedger(max_usd=0.0)
    with pytest.raises(BudgetExceededError):
        ledger.check_budget(price(paid, 1000, 100), paid)
    unknown = ProviderConfig(provider="openai-compatible", model="m")
    with pytest.raises(BudgetExceededError):
        ledger.check_budget(None, unknown)
    UsageLedger(max_usd=0.0).check_budget(0.0, ProviderConfig(provider="offline", model="x"))
    small = UsageLedger(max_usd=0.001)
    small.check_budget(price(paid, 1000, 100), paid)  # 0.00015 fits
    with pytest.raises(BudgetExceededError):
        small.check_budget(price(paid, 100_000, 0), paid)  # 0.01 does not


def test_price_unknown_is_none_not_zero() -> None:
    assert price(ProviderConfig(provider="openai-compatible", model="m"), 10, 10) is None
    assert price(ProviderConfig(provider="offline", model="m"), 10, 10) == 0.0


async def test_collect_calls_isolates_concurrent_tasks() -> None:
    ledger = UsageLedger()
    llm = OfflineLLM(ProviderConfig(provider="offline", model="g"), role="g", ledger=ledger)

    async def one(n: int) -> int:
        with collect_calls() as calls:
            for _ in range(n):
                await llm.complete([{"role": "user", "content": f"q{n}"}])
            return len(calls)

    assert await asyncio.gather(one(1), one(3), one(2)) == [1, 3, 2]
    assert len(ledger.records()) == 6


async def test_retry_async_retries_then_gives_up() -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableError("429", retry_after=0.001)
        return "ok"

    assert await retry_async(flaky, max_retries=3, what="t") == "ok"

    async def always() -> str:
        raise RetryableError("503", retry_after=0.001)

    with pytest.raises(ProviderError, match="giving up"):
        await retry_async(always, max_retries=2, what="t")


# ------------------------------------------------------------------------------- LLM providers


def _compat(handler: Any, tmp_path: Path | None = None) -> OpenAICompatLLM:
    cfg = ProviderConfig(
        provider="openai-compatible",
        model="m",
        base_url="http://x/v1",
        max_retries=2,
        price_input_per_mtok=1.0,
        price_output_per_mtok=2.0,
    )
    llm = OpenAICompatLLM(
        cfg,
        role="generator",
        ledger=UsageLedger(max_usd=1.0),
        cache=KVCache(tmp_path, "llm") if tmp_path else None,
    )
    llm._client = httpx.AsyncClient(base_url="http://x/v1", transport=httpx.MockTransport(handler))
    return llm


async def test_openai_compat_rate_limit_then_success_records_cost(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(req.content)
        assert body["temperature"] == 0 and "seed" in body
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"}, text="slow down")
        return httpx.Response(
            200,
            json={
                "model": "m-v1",
                "choices": [{"message": {"content": "Kessering [1]"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    llm = _compat(handler, tmp_path)
    res = await llm.complete([{"role": "user", "content": "hi"}], system="s")
    assert res.text == "Kessering [1]" and calls["n"] == 2
    rec = llm.ledger.records()[-1]
    assert rec.model == "m-v1" and rec.cost_usd == pytest.approx((100 + 20) / 1e6)
    again = await llm.complete([{"role": "user", "content": "hi"}], system="s")
    assert again.cached and calls["n"] == 2  # served from cache, no new request


async def test_openai_compat_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError, match="authentication"):
        await _compat(handler).complete([{"role": "user", "content": "x"}])
    assert calls["n"] == 1
    assert any(r.error for r in _compat(handler).ledger.records()) is False


async def test_openai_compat_malformed_and_network_errors() -> None:
    llm = _compat(lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="malformed"):
        await llm.complete([{"role": "user", "content": "x"}])

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="giving up"):
        await _compat(boom).complete([{"role": "user", "content": "x"}])


def test_openai_compat_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ConfigError):
        make_llm(
            ProviderConfig(provider="openai-compatible", model="m"), role="g", ledger=UsageLedger()
        )
    with pytest.raises(ConfigError):
        make_llm(ProviderConfig(provider="bogus", model="m"), role="g", ledger=UsageLedger())


async def test_anthropic_provider_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network: the SDK client is replaced; checks request shape and refusal handling."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg = ProviderConfig(
        provider="anthropic",
        model="claude-haiku-5-5",
        params={"effort": "low"},
        price_input_per_mtok=0.1,
        price_output_per_mtok=0.5,
    )
    llm = AnthropicLLM(cfg, role="extractor", ledger=UsageLedger(max_usd=1.0))
    sent: dict[str, Any] = {}

    class Block:
        type = "text"
        text = "entity<|#|>A<|#|>x<|#|>y\n<|COMPLETE|>"

    class Usage:
        input_tokens, output_tokens = 1000, 100

    class Resp:
        content = [Block()]
        usage = Usage()
        model = "claude-haiku-5-5"
        stop_reason = "end_turn"
        stop_details = None

    async def create(**kw: Any) -> Resp:
        sent.update(kw)
        return Resp()

    monkeypatch.setattr(llm._client.messages, "create", create)
    res = await llm.complete([{"role": "user", "content": "x"}], system="sys", max_tokens=50)
    assert "temperature" not in sent and sent["output_config"] == {"effort": "low"}
    assert sent["system"] == "sys" and res.prompt_tokens == 1000
    rec = llm.ledger.records()[-1]
    assert rec.cost_usd == pytest.approx((1000 * 0.1 + 100 * 0.5) / 1e6)
    assert rec.extra["sampling_controls"] == "unsupported"

    Resp.stop_reason = "refusal"
    with pytest.raises(ProviderError, match="refused"):
        await llm.complete([{"role": "user", "content": "y"}])


# ------------------------------------------------------------------------------- offline/rerank


def test_heuristic_extract_emits_lightrag_format() -> None:
    out = heuristic_extract("Ilse Brandt founded Velmora in Ostrand. Nothing else here.")
    lines = out.splitlines()
    assert lines[-1] == "<|COMPLETE|>"
    assert any(ln.startswith("entity<|#|>Ilse Brandt<|#|>") for ln in lines)
    assert any(ln.startswith("relation<|#|>Ilse Brandt<|#|>Velmora") for ln in lines)
    assert all(len(ln.split("<|#|>")) in (4, 5) for ln in lines[:-1])


def test_lexical_reranker_orders_by_relevance_and_caches(tmp_path: Path) -> None:
    rr = make_reranker(
        ProviderConfig(provider="lexical", model="l"),
        ledger=UsageLedger(),
        cache=KVCache(tmp_path, "rr"),
    )
    assert isinstance(rr, LexicalReranker)
    s = rr.score("amber harbour", ["the amber harbour", "software", "harbour"])
    assert s[0] > s[2] > s[1] == 0.0 and all(0 <= x <= 1 for x in s)
    assert rr.score("amber harbour", ["the amber harbour"]) == [s[0]]
    assert make_reranker(None, ledger=UsageLedger()) is None
    assert make_reranker(ProviderConfig(provider="none", model="-"), ledger=UsageLedger()) is None


def test_unset_env_var_in_base_url_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVATE_NOPE", raising=False)
    cfg = ProviderConfig(provider="openai-compatible", model="m", base_url="${ELEVATE_NOPE}/v1")
    with pytest.raises(ConfigError, match="not set"):
        make_llm(cfg, role="g", ledger=UsageLedger())


def test_config_hash_ignores_storage_locations(tmp_path: Path) -> None:
    from tests.conftest import offline_config

    a = offline_config(tmp_path / "laptop")
    b = offline_config(tmp_path / "ec2-volume")
    assert a.data_dir != b.data_dir
    assert a.config_hash() == b.config_hash()  # run ids survive moving the data
    assert a.config_hash() != a.model_copy(update={"dataset": "other"}).config_hash()


def test_storage_dirs_default_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = load_pipeline_config("configs/pipelines/reference.yaml").model_dump()
    for k in ("data_dir", "cache_dir", "index_dir"):
        raw.pop(k)
    monkeypatch.setenv("ELEVATE_DATA_DIR", "/mnt/elevate/data")
    monkeypatch.setenv("ELEVATE_INDEX_DIR", "/mnt/elevate/indexes")
    monkeypatch.setenv("ELEVATE_CACHE_DIR", "")  # empty means "use the default"
    cfg = PipelineConfig.model_validate(raw)
    assert (cfg.data_dir, cfg.index_dir, cfg.cache_dir) == (
        "/mnt/elevate/data",
        "/mnt/elevate/indexes",
        "data/cache",
    )
