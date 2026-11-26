"""Provider adapters for the LLM_Gateway.

``FakeLLMGateway`` lives in :mod:`elevate.gateway.providers.fake_gateway` (it imports the
gateway, which imports these adapters).
"""

from elevate.gateway.providers.anthropic import AnthropicProvider
from elevate.gateway.providers.base import (
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    PermanentProviderError,
    Prompt,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    TransientProviderError,
)
from elevate.gateway.providers.fake import FakeProvider, FakeScript, RecordReplayProvider
from elevate.gateway.providers.local import ollama_provider, vllm_provider
from elevate.gateway.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FakeProvider",
    "FakeScript",
    "Message",
    "OpenAICompatibleProvider",
    "PermanentProviderError",
    "Prompt",
    "Provider",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "RecordReplayProvider",
    "TransientProviderError",
    "ollama_provider",
    "vllm_provider",
]
