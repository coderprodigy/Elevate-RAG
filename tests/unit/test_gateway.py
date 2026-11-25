from typing import NoReturn

import pytest

from hgrag.config import AppConfig, BudgetConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.llm.budget import BudgetExceededError
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.ledger import GroupBy, UsageLedger
from hgrag.llm.providers.fake import FakeProvider
from hgrag.llm.types import LLMRequest, Message, ProviderResult


def test_first_call_is_paid_second_is_cached(gateway: LLMGateway, config: AppConfig) -> None:
    first = gateway.complete("smoke", "hello", model_key="fake-small")
    second = gateway.complete("smoke", "hello", model_key="fake-small")

    assert first.text == second.text == "FAKE[fake-small]: hello"
    assert first.cost_usd > 0 and not first.cache_hit
    assert second.cost_usd == 0 and second.cache_hit

    ledger = UsageLedger(config.paths.ledger_db)
    (row,) = ledger.summarize(GroupBy.ROLE)
    assert (row.group, row.calls, row.cache_hits) == ("smoke", 2, 1)
    assert ledger.total_cost() == pytest.approx(first.cost_usd)


def test_no_cache_forces_a_real_call(gateway: LLMGateway, fake_provider: FakeProvider) -> None:
    gateway.complete("smoke", "x", model_key="fake-small")
    gateway.complete("smoke", "x", model_key="fake-small", use_cache=False)
    assert len(fake_provider.requests) == 2


def test_role_resolves_to_configured_model(
    gateway: LLMGateway, fake_provider: FakeProvider
) -> None:
    response = gateway.complete("extractor", "hi")
    assert response.model_key == "claude-haiku-5-5"
    assert fake_provider.requests[-1].model == "claude-haiku-5-5"


def test_temperature_dropped_only_for_models_that_reject_it(
    gateway: LLMGateway, fake_provider: FakeProvider
) -> None:
    gateway.complete("smoke", "a", model_key="claude-haiku-5-5")
    gateway.complete("smoke", "a", model_key="fake-small")
    claude_request, fake_request = fake_provider.requests
    assert claude_request.temperature is None
    assert claude_request.params["thinking"] == {"type": "disabled"}
    assert fake_request.temperature == 0.0


def test_chat_messages_and_system_prompt_pass_through(
    gateway: LLMGateway, fake_provider: FakeProvider
) -> None:
    history = [Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")]
    gateway.complete("smoke", history, system="be brief", model_key="fake-small", max_tokens=7)
    request = fake_provider.requests[-1]
    assert request.messages == tuple(history)
    assert (request.system, request.max_tokens) == ("be brief", 7)


def test_budget_stops_call_before_provider_is_used(
    config: AppConfig, fake_provider: FakeProvider
) -> None:
    tiny_budget = config.model_copy(
        update={
            "budget": BudgetConfig(global_cap_usd=1e-9, per_run_cap_usd=1e-9, confirm_above_usd=0.0)
        }
    )
    gateway = build_gateway(tiny_budget, provider_factory=lambda _: fake_provider)
    with pytest.raises(BudgetExceededError):
        gateway.complete("smoke", "hello", model_key="fake-small")
    assert fake_provider.requests == []


class _BrokenProvider:
    async def complete(self, request: LLMRequest) -> NoReturn:
        raise ConnectionError("network down")


class _RefusingProvider:
    async def complete(self, request: LLMRequest) -> ProviderResult:
        return ProviderResult(text="", input_tokens=5, output_tokens=0, stop_reason="refusal")


def _gateway_with(config: AppConfig, provider: object) -> LLMGateway:
    def factory(_: ProviderSpec) -> object:
        return provider

    return build_gateway(config, RunContext(run_id="t"), provider_factory=factory)  # type: ignore[arg-type]


def test_provider_errors_are_recorded_and_reraised(config: AppConfig) -> None:
    gateway = _gateway_with(config, _BrokenProvider())
    with pytest.raises(ConnectionError):
        gateway.complete("smoke", "hello", model_key="fake-small")
    (row,) = UsageLedger(config.paths.ledger_db).summarize(GroupBy.MODEL)
    assert row.calls == 1 and row.cost_usd == 0


def test_refusals_are_flagged(config: AppConfig) -> None:
    response = _gateway_with(config, _RefusingProvider()).complete(
        "smoke", "hello", model_key="fake-small"
    )
    assert response.status == "refusal"
