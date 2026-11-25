"""Usage ledger: one SQLite row per API call (paid or cache hit), for spend tracking."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    experiment          TEXT NOT NULL,
    phase               TEXT NOT NULL,
    user                TEXT NOT NULL,
    role                TEXT NOT NULL,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    latency_ms          REAL NOT NULL,
    cache_hit           INTEGER NOT NULL,
    status              TEXT NOT NULL
)
"""


class GroupBy(StrEnum):
    """Columns that spend can be grouped by. Fixed set, so no SQL is built from user input."""

    DAY = "day"
    ROLE = "role"
    MODEL = "model"
    RUN = "run"
    EXPERIMENT = "experiment"
    PHASE = "phase"
    USER = "user"


_GROUP_SQL: dict[GroupBy, str] = {
    GroupBy.DAY: "substr(ts, 1, 10)",
    GroupBy.ROLE: "role",
    GroupBy.MODEL: "model",
    GroupBy.RUN: "run_id",
    GroupBy.EXPERIMENT: "experiment",
    GroupBy.PHASE: "phase",
    GroupBy.USER: "user",
}


@dataclass(frozen=True)
class UsageRecord:
    run_id: str
    experiment: str
    phase: str
    user: str
    role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    status: str
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class UsageSummaryRow:
    group: str
    calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class UsageLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def record(self, record: UsageRecord) -> None:
        row: dict[str, Any] = asdict(record)
        row["cache_hit"] = int(record.cache_hit)
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO usage ({columns}) VALUES ({placeholders})", tuple(row.values())
            )
            self._conn.commit()

    def total_cost(self, run_id: str | None = None) -> float:
        """Total USD spent, overall or for one run."""
        if run_id is None:
            (total,) = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM usage").fetchone()
        else:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE run_id = ?", (run_id,)
            ).fetchone()
        return float(total)

    def summarize(
        self, by: GroupBy, since: date | None = None, run_id: str | None = None
    ) -> list[UsageSummaryRow]:
        """Spend grouped by ``by``, newest/most expensive first."""
        conditions: list[str] = []
        params: list[str] = []
        if since is not None:
            conditions.append("ts >= ?")
            params.append(since.isoformat())
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        group_expr = _GROUP_SQL[by]
        query = f"""
            SELECT {group_expr} AS grp, COUNT(*), SUM(cache_hit),
                   SUM(input_tokens + cached_input_tokens), SUM(output_tokens), SUM(cost_usd)
            FROM usage {where}
            GROUP BY grp
            ORDER BY SUM(cost_usd) DESC, grp
        """
        return [
            UsageSummaryRow(str(g), int(n), int(h), int(i), int(o), float(c))
            for g, n, h, i, o, c in self._conn.execute(query, params).fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
