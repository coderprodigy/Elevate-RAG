"""Unit tests for the LLM_Gateway (Req 3.2-3.7, 3.9, 4.1-4.3, 4.7). Offline only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from elevate.config.schema import CapSettings, RetrySettings
from elevate.core.errors import BudgetExceeded, ModelCallError
from elevate.core.tokens import FallbackTokenCounter
from elevate.gateway.budget import SQLiteBudgetTracker
from elevate.gateway.cache import SQLiteCache
from elevate.gateway.gateway import (
    JSON_REREQUEST_INSTRUCTION,
    TEMPERATURE_DEVIATION,
    LLMGateway,
    RoleSettings,
    parse_json,
)
from elevate.gateway.providers.base import (
    PermanentProviderError,
    Prompt,
    TransientProviderError,
)
from elevate.gateway.providers.fake import FakeProvider
from elevate.gateway.providers.fake_gateway import FakeLLMGateway
from elevate.gateway.registry import ModelRegistry, ModelRole
from elevate.gateway.tracer import JSONLTracer


def _entry(roles: list[str], **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "verified",
        "roles": roles,
        "provider": "fake",
        "model_id": "x",
        "version": "1",
        "family": "fake",
        "context_length": 1000,
        "price_in_per_mtok": 0.0,
        "price_out_per_mtok": 0.0,
        "supports_seed": True,
    }
    body.update(overrides)
    return body


def make_registry() -> ModelRegistry:
    models = {
        "fake-embedder": _entry(["embedder"], model_id="fake-embedder"),
        "fake-reranker": _entry(["reranker"], model_id="fake-reranker"),
        "fake-llm": _entry(["extraction", "generator", "judge"], model_id="fake-llm"),
        "extractor-x": _entry(["extraction"], model_id="vendor/extractor-x", family="vx"),
        "paid-gen": _entry(
            ["generator"],
            model_id="vendor/paid-gen",
            family="vp",
            price_in_per_mtok=1_000_000.0,  # $1 per prompt token: easy cap arithmetic
            price_out_per_mtok=0.0,
        ),
        "no-temp-gen": _entry(
            ["generator"],
            model_id="vendor/no-temp",
            family="vn",
            supports_temperature=False,
            supports_seed=False,
        ),
    }
    profile = {
        "embedder": "fake-embedder",
        "reranker": "fake-reranker",
        "extraction": "fake-llm",
        "generator": "fake-llm",
        "judge": "fake-llm",
    }
    return ModelRegistry.from_mapping(
        {
            "schema_version": 1,
            "default_profile": "offline",
            "profiles": {"offline": profile},
            "models": models,
        },
        source="<test>",
    )


def gateway(
    provider: FakeProvider | None = None, tmp_path: Path | None = None, **kwargs: Any
) -> FakeLLMGateway:
    kwargs.setdefault("registry", make_registry())
    if tmp_path is not None:
        kwargs.setdefault("cache", SQLiteCache(tmp_path / "cache.sqlite3"))
    return FakeLLMGateway(provider or FakeProvider(), **kwargs)


# ------------------------------------------------------------------ routing and parameters
def test_temperature_zero_and_seed_passthrough() -> None:
    gw = gateway(seed=7)
    result = gw.complete(ModelRole.GENERATOR, "hello")
    request = gw.fake.calls[0]
    assert request.temperature == 0.0
    assert request.seed == 7
    assert request.model_id == "fake-llm"
    assert result.params["temperature"] == 0.0
    assert result.deviations == ()


def test_model_without_temperature_or_seed_omits_them_and_records_deviation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gw = gateway(roles={"generator": "no-temp-gen"}, seed=7)
    with caplog.at_level("WARNING"):
        result = gw.complete("generator", "hello")
        gw.complete("generator", "hello again")
    request = gw.fake.calls[0]
    assert request.temperature is None
    assert request.seed is None
    assert TEMPERATURE_DEVIATION in result.deviations
    assert sum(TEMPERATURE_DEVIATION in r.getMessage() for r in caplog.records) == 1


def test_extraction_and_generator_use_separate_entries_and_settings() -> None:
    gw = gateway(
        roles={"extraction": "extractor-x", "generator": "fake-llm"},
        role_settings={
            ModelRole.EXTRACTION: RoleSettings(max_tokens=3000),
            ModelRole.GENERATOR: RoleSettings(max_tokens=200),
        },
    )
    ext = gw.complete(ModelRole.EXTRACTION, "extract")
    gen = gw.complete(ModelRole.GENERATOR, "answer")
    assert (ext.model_id, gen.model_id) == ("vendor/extractor-x", "fake-llm")
    assert [c.max_tokens for c in gw.fake.calls] == [3000, 200]
    settings = gw.model_settings()
    assert settings["extraction"]["key"] == "extractor-x"
    assert settings["generator"]["max_tokens"] == 200


def test_embedder_role_is_not_an_llm_role() -> None:
    with pytest.raises(ValueError, match="not an LLM role"):
        gateway().complete(ModelRole.EMBEDDER, "x")


# ------------------------------------------------------------------ retry
def test_transient_errors_retry_with_exponential_backoff() -> None:
    provider = FakeProvider(queue=[TransientProviderError("429"), TransientProviderError("503")])
    gw = gateway(provider)
    result = gw.complete("generator", "q")
    assert result.attempts == 3
    assert gw.sleeps == [2.0, 4.0]


def test_retry_exhaustion_raises_model_call_error_naming_model_and_reason() -> None:
    provider = FakeProvider(queue=[TransientProviderError(f"overloaded {i}") for i in range(5)])
    gw = gateway(provider, retry=RetrySettings(max_attempts=5, backoff_base_s=2.0))
    with pytest.raises(ModelCallError) as info:
        gw.complete("generator", "q")
    assert info.value.model == "fake-llm"
    assert info.value.attempts == 5
    assert "overloaded 4" in str(info.value)
    assert gw.sleeps == [2.0, 4.0, 8.0, 16.0]


def test_permanent_error_is_not_retried() -> None:
    gw = gateway(FakeProvider(queue=[PermanentProviderError("HTTP 400: bad")]))
    with pytest.raises(ModelCallError, match="HTTP 400") as info:
        gw.complete("generator", "q")
    assert info.value.attempts == 1
    assert gw.sleeps == []


# ------------------------------------------------------------------ JSON handling
@pytest.mark.parametrize(
    ("text", "status"),
    [
        ('{"a": 1}', "ok"),
        ('```json\n{"a": 1,}\n```', "repaired"),
        ('Sure! {"a": 1} hope that helps', "repaired"),
        ("not json", None),
        ('{"a": NaN}', None),
    ],
)
def test_parse_json(text: str, status: str | None) -> None:
    value, got = parse_json(text)
    assert got == status
    if status is not None:
        assert value == {"a": 1}


def test_json_repair_pass_needs_no_rerequest() -> None:
    gw = gateway(FakeProvider(queue=['```json\n{"label": "correct"}\n```']))
    result = gw.complete("judge", "judge it", json=True)
    assert result.json_status == "repaired"
    assert result.parsed == {"label": "correct"}
    assert gw.provider_calls == 1


def test_json_rerequest_then_malformed() -> None:
    gw = gateway(FakeProvider(queue=["nope", '{"label": "wrong"}']))
    fixed = gw.complete("judge", "first", json=True)
    assert fixed.json_status == "rerequested"
    assert fixed.parsed == {"label": "wrong"}
    rerequest = gw.fake.calls[1]
    assert rerequest.messages[-1].content == JSON_REREQUEST_INSTRUCTION
    assert rerequest.messages[-2].content == "nope"

    gw.fake.enqueue("still not json", "also not json")
    bad = gw.complete("judge", "second", json=True)
    assert bad.malformed
    assert bad.parsed is None
    assert bad.provider_calls == 2


# ------------------------------------------------------------------ cache, budget, trace
def test_cache_hit_skips_provider_and_budget(tmp_path: Path) -> None:
    budget = SQLiteBudgetTracker("r", CapSettings(), ledger_path=tmp_path / "b.sqlite3")
    gw = gateway(
        tmp_path=tmp_path,
        budget=budget,
        roles={"generator": "paid-gen"},
        token_counter=FallbackTokenCounter(),
    )
    prompt = Prompt.from_text("one two", metadata={"question_id": "q1"})
    first = gw.complete("generator", prompt)
    second = gw.complete("generator", prompt)
    assert gw.provider_calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert first.text == second.text
    assert first.cost_usd == pytest.approx(first.prompt_tokens * 1.0)
    assert second.cost_usd == 0.0
    assert budget.run_spent_usd == pytest.approx(first.cost_usd)


def test_budget_exceeded_persists_result_in_cache(tmp_path: Path) -> None:
    caps = CapSettings(run_spending_cap_usd=1.0)
    budget = SQLiteBudgetTracker("r", caps, ledger_path=tmp_path / "b.sqlite3")
    gw = gateway(tmp_path=tmp_path, budget=budget, roles={"generator": "paid-gen"})
    with pytest.raises(BudgetExceeded):
        gw.complete("generator", "expensive prompt")
    # Rerun resumes from cache: no new provider call, no new charge, no error.
    spent = budget.run_spent_usd
    again = gw.complete("generator", "expensive prompt")
    assert again.cached
    assert gw.provider_calls == 1
    assert budget.run_spent_usd == spent
    # A new paid request is blocked before the provider is called.
    with pytest.raises(BudgetExceeded):
        gw.complete("generator", "another prompt")
    assert gw.provider_calls == 1


def test_trace_records_model_tokens_latency_cost_and_date(tmp_path: Path) -> None:
    tracer = JSONLTracer(tmp_path / "trace.jsonl")
    gw = gateway(tmp_path=tmp_path, tracer=tracer)
    result = gw.complete("generator", Prompt.from_text("hi", system="be brief"))
    gw.complete("generator", Prompt.from_text("hi", system="be brief"))
    traces = tracer.read()
    assert len(traces) == 2
    first = traces[0]
    assert first["model_id"] == "fake-llm"
    assert first["prompt_tokens"] == result.prompt_tokens > 0
    assert first["completion_tokens"] == result.completion_tokens > 0
    assert first["metadata"]["call_date"] == result.call_date
    assert first["cost_usd"] == 0.0
    assert first["input"][0] == {"role": "system", "content": "be brief"}
    assert [t["cached"] for t in traces] == [False, True]


def test_failed_call_is_traced_with_error(tmp_path: Path) -> None:
    tracer = JSONLTracer(tmp_path / "trace.jsonl")
    gw = gateway(FakeProvider(queue=[PermanentProviderError("boom")]), tracer=tracer)
    with pytest.raises(ModelCallError):
        gw.complete("generator", "q")
    (trace,) = tracer.read()
    assert "boom" in trace["error"]


def test_metadata_is_part_of_the_cache_key(tmp_path: Path) -> None:
    gw = gateway(tmp_path=tmp_path)
    gw.complete("generator", Prompt.from_text("q", metadata={"variant": "rag"}))
    gw.complete("generator", Prompt.from_text("q", metadata={"variant": "refusal"}))
    assert gw.provider_calls == 2


# ------------------------------------------------------------------ embeddings
def test_embed_caches_per_text_and_batches_misses(tmp_path: Path) -> None:
    gw = gateway(tmp_path=tmp_path)
    gw.embed_batch_size = 2
    first = gw.embed(ModelRole.EMBEDDER, ["a", "b", "c"])
    assert len(gw.fake.embed_calls) == 2  # batches of 2 + 1
    again = gw.embed_result(ModelRole.EMBEDDER, ["c", "d", "a"])
    assert again.cached_count == 2
    assert [r.texts for r in gw.fake.embed_calls[2:]] == [("d",)]
    assert again.vectors[0] == first[2]
    assert again.vectors[2] == first[0]
    assert len(first[0]) == gw.fake.embedding_dim


def test_from_config_wires_roles_seed_and_retry(tmp_path: Path) -> None:
    from elevate.config.schema import ExperimentConfig, PathSettings

    config = ExperimentConfig(
        name="t",
        seed=11,
        retry=RetrySettings(max_attempts=2, backoff_base_s=0.5),
        paths=PathSettings(cache_dir=tmp_path / "cache", trace_dir=tmp_path / "traces"),
    )
    provider = FakeProvider(queue=[TransientProviderError("x")])
    sleeps: list[float] = []
    gw = LLMGateway.from_config(
        config,
        run_id="run-1",
        registry=make_registry(),
        providers={"fake": provider},
        sleep=sleeps.append,
    )
    gw.complete("generator", "q")
    assert provider.calls[-1].seed == 11
    assert sleeps == [0.5]
    lines = (tmp_path / "traces" / "run-1.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["run_id"] == "run-1"
    assert (tmp_path / "cache" / "cache.sqlite3").is_file()
