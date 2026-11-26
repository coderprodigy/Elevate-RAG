"""``FakeLLMGateway``: the real :class:`LLMGateway` wired to a :class:`FakeProvider`.

Everything except the provider is production code (registry lookup, caching, retry, JSON
handling, budget, tracing), so offline tests exercise the same paths as real Runs
(Requirement 37.3). Retry sleeps are recorded in ``sleeps`` instead of waiting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from elevate.config.schema import RetrySettings
from elevate.core.tokens import TokenCounter
from elevate.gateway.cache import Cache
from elevate.gateway.gateway import GatewayBudget, LLMGateway, RoleSettings
from elevate.gateway.providers.fake import FakeProvider, FakeScript
from elevate.gateway.registry import ModelRegistry
from elevate.gateway.tracer import Tracer

TINY_SAMPLE_SCRIPT_DIR = Path("fixtures/tiny_sample/fake_llm")


def _fake_entry(roles: list[str], context_length: int, notes: str) -> dict[str, Any]:
    return {
        "status": "verified",
        "roles": roles,
        "provider": "fake",
        "model_id": None,  # filled in below
        "version": "1",
        "family": "fake",
        "context_length": context_length,
        "price_in_per_mtok": 0.0,
        "price_out_per_mtok": 0.0,
        "supports_seed": True,
        "notes": notes,
    }


def offline_registry() -> ModelRegistry:
    """In-memory registry with only the offline fakes (same keys as ``configs/models.yaml``)."""
    models = {
        "fake-embedder": _fake_entry(["embedder"], 8192, "Hash-based deterministic vectors."),
        "fake-reranker": _fake_entry(["reranker"], 8192, "Deterministic scores."),
        "fake-llm": _fake_entry(
            ["extraction", "generator", "judge"], 131072, "Scripted or record-replay responses."
        ),
    }
    for key, body in models.items():
        body["model_id"] = key
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
        source="<offline-registry>",
    )


class FakeLLMGateway(LLMGateway):
    """Offline gateway; ``provider_calls`` counts calls that reached the fake provider."""

    def __init__(
        self,
        provider: FakeProvider | None = None,
        *,
        script: FakeScript | None = None,
        script_dir: str | Path | None = None,
        queue: Iterable[str | BaseException] = (),
        registry: ModelRegistry | None = None,
        roles: Mapping[Any, str] | None = None,
        run_id: str = "fake-run",
        cache: Cache | None = None,
        budget: GatewayBudget | None = None,
        tracer: Tracer | None = None,
        seed: int | None = 42,
        retry: RetrySettings | None = None,
        role_settings: Mapping[Any, RoleSettings] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if provider is None:
            if script is None and script_dir is not None:
                script = FakeScript.load(script_dir)
            provider = FakeProvider(script, queue=queue)
        else:
            provider.enqueue(*queue)
        self.fake = provider
        self.sleeps: list[float] = []
        super().__init__(
            registry if registry is not None else offline_registry(),
            run_id=run_id,
            roles=roles,
            providers={"fake": provider},
            cache=cache,
            budget=budget,
            tracer=tracer,
            seed=seed,
            retry=retry,
            role_settings=role_settings,
            token_counter=token_counter,
            sleep=self.sleeps.append,
        )

    @property
    def provider_calls(self) -> int:
        return self.fake.call_count


__all__ = ["TINY_SAMPLE_SCRIPT_DIR", "FakeLLMGateway", "offline_registry"]
