"""Small shared utilities: hashing, JSONL IO, tokenisation, logging, errors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# --------------------------------------------------------------------------- errors


class ElevateError(Exception):
    """Base class for all project errors."""


class ConfigError(ElevateError):
    pass


class DataError(ElevateError):
    pass


class ProviderError(ElevateError):
    """A model provider failed (after retries) or returned an unusable response."""


class BudgetExceededError(ElevateError):
    pass


class StorageError(ElevateError):
    pass


# --------------------------------------------------------------------------- hashing


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""

    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump(mode="json")
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set | frozenset):
            return sorted(o)
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    return sha256_hex(canonical_json(obj))[:length]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_size):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- JSONL


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    """Atomically write rows to JSONL; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def repair_jsonl_tail(path: Path) -> bool:
    """Drop an incomplete trailing line left by a crash mid-write, so appends stay valid.
    Returns True if the file was modified."""
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    keep = data[: data.rfind(b"\n") + 1]  # rfind = -1 → keep nothing
    path.write_bytes(keep)
    logging.getLogger(__name__).warning("removed incomplete trailing line from %s", path)
    return True


def read_jsonl(path: Path, *, skip_corrupt_tail: bool = False) -> Iterator[dict[str, Any]]:
    """Read JSONL. With `skip_corrupt_tail`, a truncated *last* line (from a crash mid-write)
    is ignored; corruption anywhere else always raises."""
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            if skip_corrupt_tail and i == len(lines) - 1:
                logging.getLogger(__name__).warning("ignoring truncated last line in %s", path)
                return
            raise DataError(f"{path}:{i + 1}: invalid JSON: {exc}") from exc


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- tokens

DEFAULT_ENCODING = "o200k_base"  # same encoding LightRAG uses (gpt-4o-mini), see D-06


@lru_cache(maxsize=4)
def _encoding(name: str) -> Any:
    import tiktoken

    return tiktoken.get_encoding(name)


class Tokenizer:
    """Thin wrapper around tiktoken so every component counts tokens identically."""

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        self.name = encoding
        self._enc = _encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return list(self._enc.encode(text, disallowed_special=()))

    def decode(self, tokens: list[int]) -> str:
        return str(self._enc.decode(tokens))

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        toks = self.encode(text)
        return text if len(toks) <= max_tokens else self.decode(toks[:max_tokens])


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens used by BM25, lexical reranking and features."""
    return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------- logging


class _JsonFormatter(logging.Formatter):
    _SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|(?i:api[_-]?key)\s*[=:]\s*\S+)")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": self._SECRET_RE.sub("<redacted>", record.getMessage()),
        }
        for key in ("run_id", "question_id", "stage"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    if json_logs is None:
        json_logs = os.getenv("ELEVATE_JSON_LOGS", "0") == "1"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # LightRAG is chatty at INFO; keep its warnings.
    logging.getLogger("lightrag").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("nano-vectordb").setLevel(logging.WARNING)


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)
