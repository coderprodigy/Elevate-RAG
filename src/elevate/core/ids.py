"""Deterministic content-hash identifiers.

IDs depend only on content, never on time, randomness or process state, so re-running a
stage on the same input reproduces the same IDs (chunking determinism, Req 8.6) and cache
keys (Req 4.1). Multi-part IDs hash a canonical JSON encoding of the parts, so
``("ab", "c")`` and ``("a", "bc")`` never collide by concatenation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_ID_LENGTH = 16


def sha256_hex(data: str | bytes) -> str:
    """Return the full SHA-256 hex digest of ``data`` (strings are UTF-8 encoded)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to JSON with sorted keys and no insignificant whitespace.

    Raises ``TypeError`` for values that are not JSON-serialisable, rather than hashing
    an unstable ``repr``.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(obj: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON encoding of ``obj``."""
    return sha256_hex(canonical_json(obj))


def make_id(prefix: str, *parts: Any, length: int = DEFAULT_ID_LENGTH) -> str:
    """Return ``"{prefix}-{hash}"`` where ``hash`` is a truncated content hash of ``parts``."""
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if not 8 <= length <= 64:
        raise ValueError("length must be between 8 and 64 hex characters")
    return f"{prefix}-{content_hash(list(parts))[:length]}"


def doc_id(source: str, title: str, text: str) -> str:
    """ID for a Normalised_Corpus document derived from its source, title and text."""
    return make_id("doc", source, title, text)


def chunk_id(doc: str, start_offset: int, end_offset: int, text: str) -> str:
    """ID for a chunk derived from its document ID, character offsets and text."""
    return make_id("chunk", doc, start_offset, end_offset, text)


__all__ = [
    "DEFAULT_ID_LENGTH",
    "canonical_json",
    "chunk_id",
    "content_hash",
    "doc_id",
    "make_id",
    "sha256_hex",
]
