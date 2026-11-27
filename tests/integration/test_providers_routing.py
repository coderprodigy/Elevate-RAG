import asyncio
import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from elevate_rag.config import Model
from elevate_rag.data import Question
from elevate_rag.providers import Budget, BudgetExceeded, Cache, ProviderError, Providers
from elevate_rag.routing import SVM, cheapest_correct, classify_rules, cross_validate, decide


def test_cached_embeddings_have_stable_dimensions(tmp_path, settings):
    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            first = await providers.embed(["one two"])
            second = await providers.embed(["one two"])
            assert first.shape == (1, 256)
            np.testing.assert_allclose(first, second)
            assert [event.cached for event in providers.usage] == [False, True]
            query = await providers.embed(["one two"], query=True)
            np.testing.assert_allclose(first, query)
            assert providers.usage[-1].cached is False

    asyncio.run(check())


def test_budget_persists_and_cache_detects_corruption(tmp_path):
    path = tmp_path / "budget.sqlite3"
    ledger = Budget(path, 0.1)
    first = ledger.reserve(0.05)
    with pytest.raises(BudgetExceeded):
        Budget(path, 0.1).reserve(0.06)
    ledger.settle(first, 0.01)
    assert Budget(path, 0.1).reserve(0.08) > first
    cache = Cache(tmp_path / "cache")
    assert cache.get("embedding", "entry") is None
    cache.put("embedding", "entry", [1, 2])
    assert cache.get("embedding", "entry") == [1, 2]
    (tmp_path / "cache" / "embedding" / "entry.json").write_text(
        '{"checksum":"invalid","value":[1]}'
    )
    with pytest.raises(ValueError, match="checksum"):
        cache.get("embedding", "entry")


def test_provider_rate_limit_and_invalid_credentials(tmp_path, settings, monkeypatch):
    async def check():
        model = Model(
            provider="openai-compatible",
            model="api",
            revision="v1",
            base_url="https://api.example.test/v1",
            key_env="TEST_API_KEY",
            input_per_million=1,
            output_per_million=5,
            paid=True,
        )
        configured = replace(
            settings, models={**settings.models, "generation": model}, spending_cap=0.10
        )
        monkeypatch.setenv("TEST_API_KEY", "test-token")
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            assert request.headers["authorization"] == "Bearer test-token"
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            if calls == 3:
                return httpx.Response(401, text="a private API diagnostic")
            assert json.loads(request.content)["temperature"] == 0
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )

        async with Providers(
            configured, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            await providers.client.aclose()
            providers.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            assert await providers.chat("generation", "system", "question") == "ok"
            event = providers.usage[-1]
            assert event.attempts == 2
            assert event.estimated_usd == pytest.approx(20 / 1e6)
            with pytest.raises(ProviderError, match="401") as failure:
                await providers.chat("generation", "system", "different question")
            assert "private" not in str(failure.value)

    asyncio.run(check())


def test_router_with_query_types_and_offline_oracle(tmp_path, settings):
    pytest.importorskip("sklearn")
    questions = [
        Question(
            f"single-{i}",
            f"where birthplace located marker{i}",
            query_type="single-hop",
            domain="a",
        )
        for i in range(6)
    ]
    questions += [
        Question(
            f"global-{i}",
            f"summarize overall themes story marker{i}",
            query_type="global",
            domain="b",
        )
        for i in range(6)
    ]
    model = SVM.train([(q.question_id, q.text, q.query_type) for q in questions], "query-type")
    path = tmp_path / "classifier.json"
    model.save(path)
    model = SVM.read(path)
    assert model.route("where birthplace located") == "V4"
    assert model.route("summarize overall themes story") == "global"
    assert all(fold["accuracy"] == 1 for fold in cross_validate(questions, False, 5))
    duplicate = Question("duplicate", questions[0].text.upper(), query_type="single-hop")
    folds = cross_validate([*questions, duplicate], False, 5)
    assert any(
        {questions[0].question_id, duplicate.question_id} <= set(fold["test_ids"]) for fold in folds
    )
    assert classify_rules("Compare Ana and Beth", {})[0] == "mix"
    trials = [
        {"question_id": "q", "mode": "mix", "correct": True, "estimated_usd": 0.5, "tokens": 15},
        {"question_id": "q", "mode": "V4", "correct": True, "estimated_usd": 0.1, "tokens": 10},
        {
            "question_id": "q",
            "mode": "global",
            "correct": False,
            "estimated_usd": 0.01,
            "tokens": 8,
        },
    ]
    assert cheapest_correct(trials)["q"]["mode"] == "V4"

    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            route = await decide("R3", "What are the main themes?", "V4", providers)
            assert route.final == "global"
            assert providers.usage[-1].role == "router"

    asyncio.run(check())


def test_r4_calibration_and_cost_regret():
    from elevate_rag.experiments import router_metrics
    from elevate_rag.routing import calibrate_threshold

    def row(key, correct, mode, confidence, cost):
        return {
            "question_id": key,
            "status": "ok",
            "correct": correct,
            "mode": mode,
            "route": {"final": mode, "confidence": confidence},
            "answer": {"refused": False},
            "estimated_usd": cost,
            "tokens": cost * 20,
        }

    low = row("low", False, "V4", 0.2, 1)
    high = row("high", True, "V4", 0.7, 1)
    mix = [row("low", True, "mix", 0.9, 2), row("high", True, "mix", 0.9, 2)]
    calibrated = calibrate_threshold([low, high], mix)
    assert calibrated["selected"] == {"threshold": 0.7, "accuracy": 1.0, "fallback_count": 1}
    routed = [mix[0], high]
    measured = router_metrics(routed, [low, high, *mix], mix)
    assert measured["oracle_regret"] == 0
    assert measured["cost_savings_vs_mix"] == 0.25


def test_budget_configuration_disallows_remote_calls_without_prices():
    with pytest.raises(ValueError, match="spend"):
        Model("openai-compatible", "api", "revision", base_url="https://example.test/v1")
    with pytest.raises(ValueError, match="price"):
        Model(
            "openai-compatible",
            "api",
            "revision",
            base_url="https://example.test/v1",
            key_env="KEY",
            paid=True,
        )
    with pytest.raises(ValueError, match="HTTPS"):
        Model(
            "openai-compatible",
            "api",
            "revision",
            base_url="http://example.test/v1",
            key_env="KEY",
            paid=True,
        )
