"""Content-hash cache for LLM, embedding, retrieval and judge calls (Req 4.1, 4.2).

A cache key is the SHA-256 of the canonical JSON of ``(namespace, input content hash,
model ID, parameters)``, so the same request always maps to the same entry and any change
to the input, the model or a parameter produces a new one.

Values are stored as canonical JSON in a SQLite database in WAL mode, which lets several
processes and threads read and write the same file. Every value returned by
:meth:`SQLiteCache.get_or_call`, hit or miss, is the JSON round-trip of the stored value,
so the first and second call of a request return equal results (Property 1).

Concurrency: two workers that miss on the same key at the same moment may both call the
provider. The first write wins (``INSERT OR IGNORE``) and both workers return the stored
value, so results stay identical; only the duplicate spend is not prevented.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from elevate.core.ids import canonical_json, content_hash

T = TypeVar("T")
V = TypeVar("V")

CACHE_DIR_ENV = "ELEVATE_CACHE_DIR"
DEFAULT_CACHE_DIR = Path(".cache")
CACHE_FILENAME = "cache.sqlite3"
_BUSY_TIMEOUT_S = 30.0


class CacheNamespace(StrEnum):
    """Kinds of cacheable call; part of the key so namespaces never collide."""

    EXTRACTION = "extraction"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    JUDGE = "judge"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class CacheKey:
    """Identifies one cacheable request.

    ``input_hash`` is the content hash of the request input; build keys with
    :meth:`for_request` to hash the input consistently.
    """

    namespace: CacheNamespace
    input_hash: str
    model_id: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def for_request(
        cls,
        namespace: CacheNamespace | str,
        content: Any,
        model_id: str,
        params: Mapping[str, Any] | None = None,
    ) -> CacheKey:
        """Build a key from the raw request input (any JSON-serialisable value)."""
        return cls(
            namespace=CacheNamespace(namespace),
            input_hash=content_hash(content),
            model_id=model_id,
            params=dict(params or {}),
        )

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of the canonical JSON of all key fields."""
        return content_hash(
            {
                "namespace": self.namespace.value,
                "input": self.input_hash,
                "model": self.model_id,
                "params": dict(self.params),
            }
        )


@dataclass(frozen=True)
class CacheResult(Generic[V]):
    """A value plus whether it came from the cache (so only misses are charged)."""

    value: V
    hit: bool


@dataclass
class CacheStats:
    """Hit and miss counts of one cache instance, overall and per namespace."""

    hits: int = 0
    misses: int = 0
    by_namespace: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, namespace: CacheNamespace, *, hit: bool) -> None:
        bucket = self.by_namespace.setdefault(namespace.value, {"hits": 0, "misses": 0})
        if hit:
            self.hits += 1
            bucket["hits"] += 1
        else:
            self.misses += 1
            bucket["misses"] += 1


class Cache(Protocol):
    """Content-hash cache interface used by the LLM_Gateway (design: Gateway section)."""

    def get(self, key: CacheKey) -> CacheResult[Any] | None: ...

    def get_or_call(self, key: CacheKey, thunk: Callable[[], T]) -> T: ...

    def get_or_call_result(self, key: CacheKey, thunk: Callable[[], T]) -> CacheResult[T]: ...


def default_cache_path() -> Path:
    """Cache file under ``$ELEVATE_CACHE_DIR``, or under ``.cache/`` when unset."""
    configured = os.environ.get(CACHE_DIR_ENV)
    directory = Path(configured) if configured else DEFAULT_CACHE_DIR
    return directory / CACHE_FILENAME


class SQLiteCache:
    """Disk-backed :class:`Cache` stored in one SQLite file in WAL mode."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                " key TEXT PRIMARY KEY,"
                " namespace TEXT NOT NULL,"
                " model_id TEXT NOT NULL,"
                " value TEXT NOT NULL,"
                " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across threads and processes.
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S)
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def get(self, key: CacheKey) -> CacheResult[Any] | None:
        """Return the stored value for ``key``, or ``None`` on a miss. Does not touch stats."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key.digest,)).fetchone()
        if row is None:
            return None
        return CacheResult(value=json.loads(row[0]), hit=True)

    def put(self, key: CacheKey, value: Any) -> Any:
        """Store ``value`` unless the key exists; return the value now stored for ``key``.

        Raises ``TypeError``/``ValueError`` if ``value`` is not JSON-serialisable.
        """
        encoded = canonical_json(value)
        digest = key.digest
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cache (key, namespace, model_id, value) VALUES (?, ?, ?, ?)",
                (digest, key.namespace.value, key.model_id, encoded),
            )
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (digest,)).fetchone()
        return json.loads(row[0])

    def get_or_call_result(self, key: CacheKey, thunk: Callable[[], T]) -> CacheResult[T]:
        """Return the cached value, or call ``thunk`` once, store and return its result.

        ``hit`` is ``False`` exactly when ``thunk`` ran, so callers charge the budget only
        then. Exceptions from ``thunk`` propagate and nothing is stored.
        """
        cached = self.get(key)
        if cached is not None:
            self._record(key, hit=True)
            return CacheResult(value=cached.value, hit=True)
        stored = self.put(key, thunk())
        self._record(key, hit=False)
        return CacheResult(value=stored, hit=False)

    def get_or_call(self, key: CacheKey, thunk: Callable[[], T]) -> T:
        """Like :meth:`get_or_call_result` but returns only the value."""
        return self.get_or_call_result(key, thunk).value

    def __contains__(self, key: object) -> bool:
        return isinstance(key, CacheKey) and self.get(key) is not None

    def __len__(self) -> int:
        with self._connect() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return int(count)

    def _record(self, key: CacheKey, *, hit: bool) -> None:
        with self._stats_lock:
            self.stats.record(key.namespace, hit=hit)


__all__ = [
    "CACHE_DIR_ENV",
    "Cache",
    "CacheKey",
    "CacheNamespace",
    "CacheResult",
    "CacheStats",
    "SQLiteCache",
    "default_cache_path",
]
