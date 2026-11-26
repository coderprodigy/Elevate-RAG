"""Run_Record data model and Run start (Requirements 1.4, 1.6, 1.7).

:func:`open_run` is called once when a Run starts. It seeds every RNG, then captures a
secret-free configuration snapshot, the git commit hash, installed library versions and the
UTC start time into a fresh :class:`RunRecord`. Pipelines append one :class:`RunItem` per
query as the Run proceeds.

The config is accepted as any pydantic model or mapping, so this module does not depend on
the concrete ``ExperimentConfig`` schema. Redaction removes:

- ``SecretStr`` / ``SecretBytes`` values anywhere in the tree;
- string values under keys whose words include ``key``, ``token``, ``password``, ``secret``
  or similar (``api_key``, ``accessToken``, a ``secrets:`` section). The plural ``tokens``
  is deliberately not secret, so ``max_tokens`` or ``stop_tokens`` survive;
- the password part of URLs such as ``postgresql://user:pw@host/db``.
"""

from __future__ import annotations

import platform
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import Enum
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretBytes, SecretStr

from elevate.core.ids import content_hash, make_id
from elevate.core.seeding import SeedRecord, set_global_seeds

REDACTED = "**********"
UNKNOWN_COMMIT = "unknown"
DEFAULT_SEED = 42

# Key words that mark a value as secret. Matched against whole words of the key, after
# splitting on ``_``, ``-``, ``.``, spaces and camelCase boundaries.
_SECRET_WORDS = frozenset(
    {
        "key",
        "apikey",
        "apikeys",
        "token",
        "password",
        "passwords",
        "passwd",
        "pwd",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "auth",
        "authorization",
    }
)
_WORD_SPLIT = re.compile(r"[_\-.\s]+|(?<=[a-z0-9])(?=[A-Z])")
_URL_PASSWORD = re.compile(r"(?P<head>[A-Za-z][A-Za-z0-9+.\-]*://[^:/@\s]*:)[^@\s]+@")

# Distributions whose versions are recorded when installed (core first, then extras).
TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "elevate",
    "faiss-cpu",
    "nano-vectordb",
    "networkx",
    "numpy",
    "pydantic",
    "python-dotenv",
    "pyyaml",
    "rank-bm25",
    "scikit-learn",
    "scipy",
    "tiktoken",
    "typer",
    "lightrag-hku",
    "torch",
    "sentence-transformers",
    "openai",
    "anthropic",
    "qdrant-client",
    "pymilvus",
    "psycopg",
    "pgvector",
    "neo4j",
    "ragas",
    "langfuse",
)


# ---------------------------------------------------------------- models


class RunItem(BaseModel):
    """Per-query result row of a Run_Record (persisted as one JSONL line)."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    prediction: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    # Typed RouteDecision / JudgeLabel models arrive with the router and judge; until then
    # they are stored as plain JSON objects.
    route: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class RunRecord(BaseModel):
    """Persisted output of one Run: run metadata plus per-query items."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    split: str
    pipeline: str
    started_at: datetime
    config_snapshot: dict[str, Any]
    config_hash: str
    model_ids: dict[str, str] = Field(default_factory=dict)
    call_dates: list[str] = Field(default_factory=list)
    git_commit: str
    seeds: SeedRecord
    library_versions: dict[str, str]
    items: list[RunItem] = Field(default_factory=list)
    # Typed MetricsSummary arrives with the evaluation harness.
    summary: dict[str, Any] | None = None

    def add_item(self, item: RunItem) -> None:
        """Append a completed per-query result."""
        self.items.append(item)


# ---------------------------------------------------------------- redaction


def is_secret_key(key: str) -> bool:
    """Return True when any word of ``key`` names a secret (``api_key``, ``dbPassword``)."""
    words = (w.lower() for w in _WORD_SPLIT.split(key) if w)
    return any(w in _SECRET_WORDS for w in words)


def redact_url_password(text: str) -> str:
    """Replace the password in ``scheme://user:password@host`` with :data:`REDACTED`."""
    return _URL_PASSWORD.sub(lambda m: f"{m.group('head')}{REDACTED}@", text)


def _redact(value: Any, *, secret: bool) -> Any:
    if isinstance(value, SecretStr | SecretBytes):
        return REDACTED
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="python"), secret=secret)
    if isinstance(value, Mapping):
        return {
            str(k): _redact(v, secret=secret or is_secret_key(str(k))) for k, v in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
        return [_redact(v, secret=secret) for v in items]
    if isinstance(value, bytes):
        return REDACTED if secret and value else value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        if secret and value:
            return REDACTED
        return redact_url_password(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Enum):
        return _redact(value.value, secret=secret)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    return _redact(str(value), secret=secret)


def redact_config(config: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of ``config`` with every secret value replaced by REDACTED."""
    if not isinstance(config, BaseModel | Mapping):
        raise TypeError(f"config must be a pydantic model or mapping, got {type(config).__name__}")
    result = _redact(config, secret=False)
    assert isinstance(result, dict)
    return result


# ---------------------------------------------------------------- environment capture


def git_commit_hash(repo_dir: str | Path | None = None) -> str:
    """Return ``HEAD``'s commit hash (``-dirty`` suffix for uncommitted changes).

    Returns :data:`UNKNOWN_COMMIT` when git is missing, ``repo_dir`` is not a repository,
    or the repository has no commits yet.
    """
    cwd = Path(repo_dir) if repo_dir is not None else None

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = git("rev-parse", "--verify", "HEAD")
    if not commit:
        return UNKNOWN_COMMIT
    status = git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if status else commit


def library_versions(distributions: tuple[str, ...] = TRACKED_DISTRIBUTIONS) -> dict[str, str]:
    """Return installed versions of ``distributions`` plus the Python version."""
    versions = {"python": platform.python_version()}
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


# ---------------------------------------------------------------- run start


def _lookup(snapshot: Mapping[str, Any], *paths: str) -> Any:
    """Return the first non-None value at any dotted ``path`` in ``snapshot``."""
    for path in paths:
        node: Any = snapshot
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


def open_run(
    config: BaseModel | Mapping[str, Any],
    *,
    dataset: str | None = None,
    split: str | None = None,
    pipeline: str | None = None,
    seed: int | None = None,
    model_ids: Mapping[str, str] | None = None,
    repo_dir: str | Path | None = None,
    started_at: datetime | None = None,
) -> RunRecord:
    """Start a Run: seed all RNGs and capture the reproducibility metadata.

    ``dataset``, ``split``, ``pipeline`` and ``seed`` default to the config's top-level (or
    ``run.``-nested) fields of the same name; ``seed`` falls back to :data:`DEFAULT_SEED`.
    """
    snapshot = redact_config(config)

    seed_value = seed if seed is not None else _lookup(snapshot, "seed", "run.seed", "seeds.global")
    seeds = set_global_seeds(DEFAULT_SEED if seed_value is None else int(seed_value))

    def field(given: str | None, name: str) -> str:
        if given is not None:
            return given
        found = _lookup(snapshot, name, f"run.{name}")
        return str(found) if found is not None else "unspecified"

    dataset_name = field(dataset, "dataset")
    split_name = field(split, "split")
    pipeline_name = field(pipeline, "pipeline")

    start = started_at or datetime.now(UTC)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    config_hash = content_hash(snapshot)
    run_id = make_id(
        f"run-{pipeline_name}-{dataset_name}-{split_name}",
        config_hash,
        seeds.seed,
        start.isoformat(),
    )

    return RunRecord(
        run_id=run_id,
        dataset=dataset_name,
        split=split_name,
        pipeline=pipeline_name,
        started_at=start,
        config_snapshot=snapshot,
        config_hash=config_hash,
        model_ids=dict(model_ids or {}),
        git_commit=git_commit_hash(repo_dir),
        seeds=seeds,
        library_versions=library_versions(),
    )


__all__ = [
    "DEFAULT_SEED",
    "REDACTED",
    "TRACKED_DISTRIBUTIONS",
    "UNKNOWN_COMMIT",
    "RunItem",
    "RunRecord",
    "git_commit_hash",
    "is_secret_key",
    "library_versions",
    "open_run",
    "redact_config",
    "redact_url_password",
]
