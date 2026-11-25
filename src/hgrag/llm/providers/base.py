"""The interface every provider adapter implements."""

from __future__ import annotations

import os
from typing import Protocol

from hgrag.llm.types import LLMRequest, ProviderResult


class ProviderConfigError(RuntimeError):
    """A provider cannot be used: missing API key, base URL, etc."""


class Provider(Protocol):
    async def complete(self, request: LLMRequest) -> ProviderResult: ...


def read_env(name: str | None) -> str | None:
    """Value of environment variable ``name`` (None if unset or blank)."""
    if name is None:
        return None
    value = os.environ.get(name, "").strip()
    return value or None


def require_env(name: str | None, *, purpose: str) -> str:
    value = read_env(name)
    if value is None:
        raise ProviderConfigError(f"{purpose}: set {name} in .env (see .env.example)")
    return value
