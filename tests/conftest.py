"""Shared fixtures: an isolated config that writes its SQLite files under a temp directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from hgrag.config import AppConfig, ProviderSpec, load_config
from hgrag.core.runs import RunContext
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.providers.fake import FakeProvider


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """The real project config, with all output paths redirected to ``tmp_path``."""
    real = load_config()
    paths = real.paths.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "runs_dir": tmp_path / "runs",
            "cache_db": tmp_path / "cache.sqlite",
            "ledger_db": tmp_path / "usage.sqlite",
        }
    )
    return real.model_copy(update={"paths": paths})


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def gateway(config: AppConfig, fake_provider: FakeProvider) -> LLMGateway:
    """A gateway whose every provider is the in-memory fake (no network, no keys)."""

    def factory(_: ProviderSpec) -> FakeProvider:
        return fake_provider

    return build_gateway(
        config, RunContext(run_id="test-run", user="tester"), provider_factory=factory
    )
