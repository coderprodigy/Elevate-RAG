"""Shared fixtures. Every fixture is offline and deterministic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elevate_rag.config import PipelineConfig, ProviderConfig
from elevate_rag.data.synthetic import write_synthetic
from elevate_rag.providers.base import UsageLedger
from elevate_rag.providers.llm import BaseLLM, LLMResult

REPO = Path(__file__).resolve().parents[1]


def offline_config(tmp: Path, **updates: Any) -> PipelineConfig:
    data = {
        "name": "test-offline",
        "dataset": "synthetic",
        "demo": True,
        "models": {
            "embedder": {
                "provider": "hash",
                "model": "hash-256",
                "dimension": 256,
                "family": "offline",
            },
            "reranker": {"provider": "lexical", "model": "lex", "family": "offline"},
            "extractor": {"provider": "offline", "model": "heur", "family": "offline"},
            "keyword": {"provider": "offline", "model": "kw", "family": "offline"},
            "generator": {"provider": "offline", "model": "extractive", "family": "offline"},
            "router_llm": {"provider": "offline", "model": "router", "family": "offline"},
            "judge": {"provider": "offline", "model": "judge", "family": "offline-judge"},
        },
        "retrieval": {"candidates": 20, "final_k": 5, "context_budget": 2048},
        "resolution": {"enabled": True, "similarity_threshold": 0.97},
        "data_dir": str(tmp / "data"),
        "cache_dir": str(tmp / "cache"),
        "index_dir": str(tmp / "indexes"),
    }
    for k, v in updates.items():
        data[k] = v
    return PipelineConfig.model_validate(data)


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> PipelineConfig:
    cfg = offline_config(tmp_path)
    write_synthetic(Path(cfg.data_dir))
    return cfg


class ScriptedLLM(BaseLLM):
    """Returns queued responses in order (for judge/router/provider-contract tests)."""

    def __init__(self, responses: list[str], **kw: Any) -> None:
        cfg = ProviderConfig(provider="offline", model="scripted", family="test")
        super().__init__(cfg, role=kw.pop("role", "test"), ledger=kw.pop("ledger", UsageLedger()))
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def _call(self, messages: Any, system: Any, max_tokens: int, hints: Any) -> LLMResult:
        self.prompts.append(messages[-1]["content"])
        text = self.responses.pop(0) if self.responses else ""
        return LLMResult(
            text=text, model="scripted", prompt_tokens=10, completion_tokens=5, latency_s=0.0
        )


@pytest.fixture
def scripted() -> type[ScriptedLLM]:
    return ScriptedLLM
