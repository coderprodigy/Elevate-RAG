"""Unit tests for the Tracer (Req 4.6, 4.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from elevate.config.schema import Endpoints, ExperimentConfig, PathSettings, Secrets
from elevate.gateway.tracer import JSONLTracer, LangfuseTracer, LLMTrace, make_tracer


def config(tmp_path: Path, *, keys: bool) -> ExperimentConfig:
    secrets = (
        Secrets(
            langfuse_public_key=SecretStr("pk-lf-test"),
            langfuse_secret_key=SecretStr("sk-lf-test"),
        )
        if keys
        else Secrets()
    )
    return ExperimentConfig(
        name="t",
        paths=PathSettings(trace_dir=tmp_path / "traces"),
        secrets=secrets,
        endpoints=Endpoints(langfuse_host="http://localhost:3000"),
    )


def call(**overrides: Any) -> LLMTrace:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "role": "generator",
        "model_key": "fake-llm",
        "model_id": "fake-llm-v1",
        "input": [{"role": "user", "content": "hi"}],
        "output": "hello",
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "cost_usd": 0.0,
    }
    values.update(overrides)
    return LLMTrace(**values)


class Observation:
    def __init__(self) -> None:
        self.ended = False

    def end(self) -> None:
        self.ended = True


class RecordingClient:
    """Stands in for ``langfuse.Langfuse``; records the observations it is sent."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[dict[str, Any]] = []
        self.observations: list[Observation] = []
        self.flushed = False

    def start_observation(self, **kwargs: Any) -> Observation:
        if self.fail:
            raise ConnectionError("unreachable")
        self.sent.append(kwargs)
        obs = Observation()
        self.observations.append(obs)
        return obs

    def flush(self) -> None:
        self.flushed = True


def test_no_credentials_writes_local_jsonl(tmp_path: Path) -> None:
    tracer = make_tracer(config(tmp_path, keys=False), "run-1")
    assert isinstance(tracer, JSONLTracer)
    tracer.trace(call())
    tracer.trace(call(output="again", cached=True))
    tracer.flush()
    assert tracer.path == tmp_path / "traces" / "run-1.jsonl"
    rows = tracer.read()
    assert [r["output"] for r in rows] == ["hello", "again"]
    assert rows[0]["model_id"] == "fake-llm-v1"
    assert rows[0]["prompt_tokens"] == 3
    assert rows[1]["cached"] is True
    assert "timestamp" in rows[0]


def test_credentials_send_every_call_to_langfuse(tmp_path: Path) -> None:
    client = RecordingClient()
    seen: dict[str, Any] = {}

    def factory(secrets: Secrets, endpoints: Endpoints) -> RecordingClient:
        seen["host"] = endpoints.langfuse_host
        return client

    tracer = make_tracer(config(tmp_path, keys=True), "run-1", client_factory=factory)
    assert isinstance(tracer, LangfuseTracer)
    tracer.trace(call(cost_usd=0.25))
    tracer.trace(call(error="boom"))
    tracer.flush()
    assert seen["host"] == "http://localhost:3000"
    assert len(client.sent) == 2
    first = client.sent[0]
    assert first["as_type"] == "generation"
    assert first["model"] == "fake-llm-v1"
    assert first["usage_details"] == {"input": 3, "output": 1}
    assert first["cost_details"] == {"total": 0.25}
    assert first["metadata"]["run_id"] == "run-1"
    assert client.sent[1]["level"] == "ERROR"
    assert all(o.ended for o in client.observations)
    assert client.flushed
    assert not tracer.fallback.path.exists()


def test_failed_langfuse_send_falls_back_to_jsonl(tmp_path: Path) -> None:
    tracer = make_tracer(
        config(tmp_path, keys=True),
        "run-1",
        client_factory=lambda s, e: RecordingClient(fail=True),
    )
    assert isinstance(tracer, LangfuseTracer)
    tracer.trace(call())
    assert [r["output"] for r in tracer.fallback.read()] == ["hello"]


def test_missing_langfuse_package_falls_back_to_jsonl(tmp_path: Path) -> None:
    def factory(secrets: Secrets, endpoints: Endpoints) -> Any:
        raise ImportError("No module named 'langfuse'")

    tracer = make_tracer(config(tmp_path, keys=True), "run-1", client_factory=factory)
    assert isinstance(tracer, JSONLTracer)


def test_default_factory_without_package_uses_jsonl(tmp_path: Path) -> None:
    try:
        import langfuse  # noqa: F401
    except ImportError:
        tracer = make_tracer(config(tmp_path, keys=True), "run-1")
        assert isinstance(tracer, JSONLTracer)
    else:
        pytest.skip("langfuse installed; covered by integration tests")


def test_traces_never_contain_secret_values(tmp_path: Path) -> None:
    client = RecordingClient()
    tracer = make_tracer(config(tmp_path, keys=True), "run-1", client_factory=lambda s, e: client)
    assert isinstance(tracer, LangfuseTracer)
    tracer.trace(call())
    tracer.fallback.trace(call())
    assert "sk-lf-test" not in repr(client.sent)
    assert "sk-lf-test" not in tracer.fallback.path.read_text(encoding="utf-8")
