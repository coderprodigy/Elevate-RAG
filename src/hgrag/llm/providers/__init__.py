"""Provider adapters. This package is the only place allowed to import vendor SDKs."""

from __future__ import annotations

from hgrag.config import ProviderSpec
from hgrag.llm.providers.base import Provider, ProviderConfigError


def build_provider(spec: ProviderSpec) -> Provider:
    """Create the adapter for ``spec``. Vendor SDKs are imported only when actually used."""
    if spec.kind == "anthropic":
        from hgrag.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec)
    if spec.kind == "openai_compat":
        from hgrag.llm.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(spec)
    from hgrag.llm.providers.fake import FakeProvider

    return FakeProvider()


__all__ = ["Provider", "ProviderConfigError", "build_provider"]
