"""Token counting with tiktoken and a deterministic offline fallback.

``get_token_counter()`` returns a :class:`TiktokenCounter` when the BPE file for the
encoding can be loaded, otherwise a :class:`FallbackTokenCounter`. tiktoken downloads its
BPE file on first use with no network timeout, so in offline mode we only use tiktoken if
the file is already in its local cache. Counts from the two backends differ; record
``counter.name`` with any result that depends on token counts.

Backend selection is controlled by ``ELEVATE_TOKENIZER``:

- ``auto`` (default): tiktoken if loadable, else fallback. Offline mode
  (``ELEVATE_OFFLINE=1`` or ``HF_HUB_OFFLINE=1``) never triggers a download.
- ``tiktoken``: require tiktoken; raise :class:`ConfigError` if it cannot load.
- ``fallback``: always use the deterministic fallback.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from elevate.core.errors import ConfigError

DEFAULT_ENCODING = "cl100k_base"
TOKENIZER_ENV = "ELEVATE_TOKENIZER"
_OFFLINE_ENVS = ("ELEVATE_OFFLINE", "HF_HUB_OFFLINE")
_TIKTOKEN_BLOB_URLS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}

# Fallback tokens: runs of letters/digits cut into pieces of at most 4 characters (a rough
# BPE approximation), or single non-space symbols. Whitespace is never a token.
_FALLBACK_PATTERN = re.compile(r"[^\W_]{1,4}|[^\w\s]|_")


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens and truncates text to a token budget."""

    @property
    def name(self) -> str:
        """Backend identifier recorded alongside token-dependent results."""
        ...

    def count(self, text: str) -> int:
        """Number of tokens in ``text``."""
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Longest prefix of ``text`` (by the backend's token boundaries) with at most
        ``max_tokens`` tokens."""
        ...


class FallbackTokenCounter:
    """Deterministic, dependency-free token counter used when tiktoken is unavailable."""

    name = "fallback-regex-v1"

    def count(self, text: str) -> int:
        return sum(1 for _ in _FALLBACK_PATTERN.finditer(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        _check_budget(max_tokens)
        if max_tokens == 0:
            return ""
        end = 0
        for i, match in enumerate(_FALLBACK_PATTERN.finditer(text)):
            if i == max_tokens:
                return text[:end]
            end = match.end()
        return text


class TiktokenCounter:
    """Token counter backed by a tiktoken encoding."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken-{encoding_name}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        _check_budget(max_tokens)
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        # A cut can split a multi-byte character; drop tokens until the decoded prefix
        # is clean and re-encodes within budget.
        k = max_tokens
        while k > 0:
            prefix = self._encoding.decode(tokens[:k], errors="replace")
            if text.startswith(prefix) and self.count(prefix) <= max_tokens:
                return prefix
            k -= 1
        return ""


def _check_budget(max_tokens: int) -> None:
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")


def _is_offline() -> bool:
    return any(
        os.environ.get(var, "").strip().lower() in {"1", "true", "yes"} for var in _OFFLINE_ENVS
    )


def _tiktoken_cached(encoding_name: str) -> bool:
    """Whether tiktoken's BPE file for ``encoding_name`` is already in its local cache.

    Mirrors tiktoken's cache layout: ``sha1(blob_url)`` under ``TIKTOKEN_CACHE_DIR``,
    ``DATA_GYM_CACHE_DIR`` or ``<tmp>/data-gym-cache``.
    """
    url = _TIKTOKEN_BLOB_URLS.get(encoding_name)
    if url is None:
        return False
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
    base = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "data-gym-cache"
    return (base / hashlib.sha1(url.encode()).hexdigest()).is_file()


@functools.cache
def _load_counter(mode: str, encoding_name: str, offline: bool) -> TokenCounter:
    if mode == "fallback":
        return FallbackTokenCounter()
    if mode not in {"auto", "tiktoken"}:
        raise ConfigError(
            file=f"${TOKENIZER_ENV}",
            field=TOKENIZER_ENV,
            expected="one of 'auto', 'tiktoken', 'fallback'",
            detail=f"got {mode!r}",
        )
    if mode == "auto" and offline and not _tiktoken_cached(encoding_name):
        return FallbackTokenCounter()
    try:
        return TiktokenCounter(encoding_name)
    except Exception as exc:
        if mode == "tiktoken":
            raise ConfigError(
                file=f"${TOKENIZER_ENV}",
                field=TOKENIZER_ENV,
                expected=f"a loadable tiktoken encoding '{encoding_name}'",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return FallbackTokenCounter()


def get_token_counter(encoding_name: str = DEFAULT_ENCODING) -> TokenCounter:
    """Return the process-wide token counter for the current environment settings."""
    mode = os.environ.get(TOKENIZER_ENV, "auto").strip().lower() or "auto"
    return _load_counter(mode, encoding_name, _is_offline())


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` with the default counter."""
    return get_token_counter().count(text)


__all__ = [
    "DEFAULT_ENCODING",
    "TOKENIZER_ENV",
    "FallbackTokenCounter",
    "TiktokenCounter",
    "TokenCounter",
    "count_tokens",
    "get_token_counter",
]
