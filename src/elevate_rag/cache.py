"""Content-hash keyed SQLite cache (D-10).

Every value is stored with a checksum; a row that fails to decode or verify is treated as a miss,
deleted and recomputed (stale/corrupt cache handling). Thread-safe: one connection per thread.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from elevate_rag.utils import canonical_json, sha256_hex

log = logging.getLogger(__name__)


class KVCache:
    def __init__(self, path: Path | str | None, namespace: str) -> None:
        self.namespace = namespace
        self.enabled = path is not None
        self._path = Path(path) / f"{namespace}.sqlite" if path is not None else None
        self._local = threading.local()
        self.hits = 0
        self.misses = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL,"
                    " sum TEXT NOT NULL)"
                )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            assert self._path is not None
            conn = sqlite3.connect(self._path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def key(payload: Any) -> str:
        return sha256_hex(canonical_json(payload))

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        row = self._conn().execute("SELECT v, sum FROM kv WHERE k=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        value, checksum = row
        try:
            if sha256_hex(value) != checksum:
                raise ValueError("checksum mismatch")
            decoded = json.loads(value)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("corrupt cache entry in %s (%s); recomputing", self.namespace, exc)
            self.delete(key)
            self.misses += 1
            return None
        self.hits += 1
        return decoded

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        encoded = json.dumps(value, ensure_ascii=False)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO kv (k, v, sum) VALUES (?, ?, ?)",
                (key, encoded, sha256_hex(encoded)),
            )

    def delete(self, key: str) -> None:
        if self.enabled:
            with self._conn() as c:
                c.execute("DELETE FROM kv WHERE k=?", (key,))

    def __len__(self) -> int:
        if not self.enabled:
            return 0
        return int(self._conn().execute("SELECT COUNT(*) FROM kv").fetchone()[0])

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
