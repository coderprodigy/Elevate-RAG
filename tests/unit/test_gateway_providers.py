"""Unit tests for provider adapters and the scripted fake (Req 3.3, 3.9, 37.3). No network."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from elevate.config.schema import ExperimentConfig, Secrets
from elevate.core.errors import ModelCallError
from elevate.gateway.gateway import default_provider_factory
from elevate.gateway.providers.anthropic import AnthropicProvider
from elevate.gateway.providers.base import (
    EmbeddingRequest,
    Message,
    PermanentProviderError,
    Prompt,
    ProviderRequest,
    TransientProviderError,
)
from elevate.gateway.providers.fake import (
    REFUSAL_STRING,
    FakeProvider,
    FakeScript,
    RecordReplayProvider,
    render_citations,
)
from elevate.gateway.providers.fake_gateway import FakeLLMGateway
from elevate.gateway.providers.local import ollama_provider, vllm_provider
from elevate.gateway.providers.openai_compat import OpenAICompatibleProvider
from elevate.gateway.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "fixtures" / "tiny_sample" / "fake_llm"
REGISTRY_FILE = ROOT / "configs" / "models.yaml"


def request(**overrides: Any) -> ProviderRequest:
    fields: dict[str, Any] = {
        "role": "generator",
        "model_id": "m",
        "messages": (Message("system", "sys"), Message("user", "hi")),
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 3,
    }
    fields.update(overrides)
    return ProviderRequest(**fields)


class RecordingTransport:
    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
    ) -> tuple[int, Any]:
        self.calls.append((url, dict(headers), dict(payload)))
        return self.responses.pop(0)


# ------------------------------------------------------------------ OpenAI-compatible
def test_openai_payload_has_temperature_seed_json_and_parses_usage() -> None:
    transport = RecordingTransport(
        (
            200,
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    provider = OpenAICompatibleProvider(api_key="sk-test", transport=transport)
    response = provider.complete(request(json_mode=True))
    url, headers, payload = transport.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 3
    assert payload["max_completion_tokens"] == 64
    assert payload["response_format"] == {"type": "json_object"}
    assert (response.text, response.prompt_tokens, response.completion_tokens) == ("ok", 5, 1)


def test_openai_payload_omits_unsupported_parameters() -> None:
    provider = OpenAICompatibleProvider(transport=RecordingTransport())
    payload = provider.chat_payload(request(temperature=None, seed=None))
    assert "temperature" not in payload
    assert "seed" not in payload


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, TransientProviderError),
        (503, TransientProviderError),
        (400, PermanentProviderError),
        (401, PermanentProviderError),
    ],
)
def test_openai_http_errors_are_classified(status: int, error: type[Exception]) -> None:
    transport = RecordingTransport((status, {"error": {"message": "nope"}}))
    provider = OpenAICompatibleProvider(transport=transport)
    with pytest.raises(error, match=f"HTTP {status}: nope"):
        provider.complete(request())


def test_openai_embeddings_sorted_by_index() -> None:
    body = {
        "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}],
        "usage": {"prompt_tokens": 4},
    }
    provider = OpenAICompatibleProvider(transport=RecordingTransport((200, body)))
    response = provider.embed(EmbeddingRequest(role="embedder", model_id="e", texts=("a", "b")))
    assert response.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert response.prompt_tokens == 4


def test_non_http_base_url_is_rejected() -> None:
    with pytest.raises(PermanentProviderError, match="http"):
        OpenAICompatibleProvider(base_url="file:///etc/passwd")


def test_local_providers_use_v1_and_max_tokens() -> None:
    transport = RecordingTransport(
        (200, {"choices": [{"message": {"content": "a"}}]}),
        (200, {"choices": [{"message": {"content": "b"}}]}),
    )
    ollama_provider("http://localhost:11434", transport=transport).complete(request())
    vllm_provider("http://gpu:8000/v1", "key", transport=transport).complete(request())
    (u1, h1, p1), (u2, h2, p2) = transport.calls
    assert u1 == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in h1
    assert u2 == "http://gpu:8000/v1/chat/completions"
    assert h2["Authorization"] == "Bearer key"
    assert p1["max_tokens"] == p2["max_tokens"] == 64


# ------------------------------------------------------------------ Anthropic
class _FakeMessages:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_anthropic_splits_system_and_omits_missing_temperature() -> None:
    reply = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text="hmm"),
            SimpleNamespace(type="text", text="hi"),
        ],
        usage=SimpleNamespace(input_tokens=9, output_tokens=2),
        stop_reason="end_turn",
    )
    messages = _FakeMessages(reply)
    provider = AnthropicProvider(client=SimpleNamespace(messages=messages))
    response = provider.complete(request(temperature=None, seed=None))
    assert messages.kwargs["system"] == "sys"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in messages.kwargs
    assert "seed" not in messages.kwargs
    assert (response.text, response.prompt_tokens, response.completion_tokens) == ("hi", 9, 2)


def test_anthropic_status_errors_are_classified() -> None:
    class RateLimitError(Exception):
        status_code = 429

    provider = AnthropicProvider(client=SimpleNamespace(messages=_FakeMessages(RateLimitError())))
    with pytest.raises(TransientProviderError, match="429"):
        provider.complete(request())


def test_anthropic_without_key_fails_clearly() -> None:
    with pytest.raises(PermanentProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider().complete(request())


# ------------------------------------------------------------------ provider factory
def test_factory_reads_secrets_and_requires_openai_key() -> None:
    registry = ModelRegistry.load(REGISTRY_FILE)
    entry = registry.get("qwen3.6-35b-a3b")
    factory = default_provider_factory(ExperimentConfig(name="t"), environ={})
    provider = factory(entry)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "vllm"

    openai_entry = entry.model_copy(update={"provider": "openai", "api_key_env": "OPENAI_API_KEY"})
    with pytest.raises(PermanentProviderError, match="OPENAI_API_KEY is not set"):
        factory(openai_entry)
    with_key = ExperimentConfig(name="t", secrets=Secrets(openai_api_key="sk-x"))
    assert default_provider_factory(with_key, environ={})(openai_entry).name == "openai"


def test_missing_adapter_becomes_model_call_error() -> None:
    registry = ModelRegistry.load(REGISTRY_FILE)
    gw = FakeLLMGateway(registry=registry, roles={"embedder": "bge-m3"})
    with pytest.raises(ModelCallError, match="sentence-transformers"):
        gw.embed("embedder", ["x"])


# ------------------------------------------------------------------ tiny-sample scripts
@pytest.fixture(scope="module")
def script() -> FakeScript:
    return FakeScript.load(SCRIPT_DIR)


def test_render_citations_by_context_order() -> None:
    template = "A [[d2]] and B [[d1]] [[d9]]."
    assert render_citations(template, ["d1", "d1", "d2"]) == "A [3] and B [1]."
    assert render_citations(template, None) == "A and B."


def test_scripted_extraction_matches_first_sentence(script: FakeScript) -> None:
    gw = FakeLLMGateway(script=script)
    first = script.extraction[0]
    prompt = f"Extract entities.\n\n{first['match']} and more text."
    result = gw.complete("extraction", prompt)
    assert result.text == first["response"]
    assert gw.complete("extraction", "unrelated text").text == "<|COMPLETE|>"


def test_scripted_generation_renders_and_refuses(script: FakeScript) -> None:
    gw = FakeLLMGateway(script=script)
    meta = {"question_id": "tiny-q01", "variant": "rag"}
    hit = gw.complete(
        "generator",
        Prompt.from_text("Q", metadata={**meta, "context_doc_ids": ["tiny-doc-07", "tiny-doc-01"]}),
    )
    assert hit.text == "1987 [2]"
    miss = gw.complete(
        "generator", Prompt.from_text("Q", metadata={**meta, "context_doc_ids": ["tiny-doc-07"]})
    )
    assert miss.text == REFUSAL_STRING
    closed = gw.complete(
        "generator",
        Prompt.from_text("Q", metadata={"question_id": "tiny-q01", "variant": "closed_book"}),
    )
    assert closed.text == "1975"


def test_scripted_judge_and_fallback(script: FakeScript) -> None:
    gw = FakeLLMGateway(script=script)
    scripted = gw.complete(
        "judge",
        Prompt.from_text("J", metadata={"question_id": "tiny-q01", "prediction": "1987 [2]"}),
        json=True,
    )
    assert scripted.parsed["label"] == "correct"
    fallback = gw.complete(
        "judge",
        Prompt.from_text(
            "J",
            metadata={"question_id": "tiny-q01", "prediction": "The 1987.", "answers": ["1987"]},
        ),
        json=True,
    )
    assert fallback.parsed["label"] == "correct"
    null_q = gw.complete(
        "judge",
        Prompt.from_text(
            "J", metadata={"question_id": "x", "prediction": "42", "answerable": False}
        ),
        json=True,
    )
    assert null_q.parsed["label"] == "wrong"


def test_fake_embeddings_are_deterministic_unit_vectors() -> None:
    provider = FakeProvider(embedding_dim=16)
    req = EmbeddingRequest(role="embedder", model_id="e", texts=("a", "b"))
    v1 = provider.embed(req).vectors
    assert v1 == provider.embed(req).vectors
    assert v1[0] != v1[1]
    assert sum(x * x for x in v1[0]) == pytest.approx(1.0)


# ------------------------------------------------------------------ record-replay
def test_record_then_replay_offline(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    inner = FakeProvider(queue=["recorded answer"])
    recorder = RecordReplayProvider(path, inner, mode="record")
    assert recorder.complete(request()).text == "recorded answer"
    recorder.embed(EmbeddingRequest(role="embedder", model_id="e", texts=("a",)))

    replay = RecordReplayProvider(path)
    assert len(replay) == 2
    assert replay.complete(request()).text == "recorded answer"
    assert replay.embed(EmbeddingRequest(role="embedder", model_id="e", texts=("a",))).vectors
    with pytest.raises(PermanentProviderError, match="no recorded response"):
        replay.complete(request(seed=4))
    assert all(json.loads(line)["key"] for line in path.read_text().splitlines())


def test_record_mode_needs_inner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inner provider"):
        RecordReplayProvider(tmp_path / "r.jsonl", mode="record")
