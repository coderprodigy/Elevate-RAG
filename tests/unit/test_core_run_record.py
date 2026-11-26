from __future__ import annotations

import json
import random
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, SecretStr

from elevate.core.run_record import (
    DEFAULT_SEED,
    REDACTED,
    UNKNOWN_COMMIT,
    RunItem,
    RunRecord,
    git_commit_hash,
    is_secret_key,
    library_versions,
    open_run,
    redact_config,
)
from elevate.core.seeding import SeedRecord, set_global_seeds

# ---------------------------------------------------------------- seeding


def test_set_global_seeds_makes_python_and_numpy_reproducible():
    rec = set_global_seeds(123)
    first = (random.random(), np.random.rand())
    set_global_seeds(123)
    assert (random.random(), np.random.rand()) == first
    assert rec.seed == rec.python == rec.numpy == 123


def test_set_global_seeds_records_torch_only_when_installed():
    rec = set_global_seeds(7)
    try:
        import torch  # noqa: F401
    except ImportError:
        assert rec.torch is None
    else:
        assert rec.torch == 7


@pytest.mark.parametrize("bad", [-1, 2**32, True, 1.5])
def test_set_global_seeds_rejects_invalid_seeds(bad):
    with pytest.raises(ValueError):
        set_global_seeds(bad)


# ---------------------------------------------------------------- redaction


class _Db(BaseModel):
    url: str
    password: SecretStr


class _Cfg(BaseModel):
    seed: int
    dataset: str
    max_tokens: int
    openai_api_key: SecretStr
    db: _Db
    paths: list[Path]


def _cfg() -> _Cfg:
    return _Cfg(
        seed=11,
        dataset="hotpotqa",
        max_tokens=512,
        openai_api_key=SecretStr("sk-live-abc"),
        db=_Db(url="postgresql://elevate:hunter2@localhost/db", password=SecretStr("hunter2")),
        paths=[Path("data/raw")],
    )


@pytest.mark.parametrize(
    ("key", "secret"),
    [
        ("api_key", True),
        ("OPENAI_API_KEY", True),
        ("accessToken", True),
        ("db_password", True),
        ("client-secret", True),
        ("max_tokens", False),
        ("prompt_tokens", False),
        ("keyword", False),
        ("dataset", False),
    ],
)
def test_is_secret_key(key, secret):
    assert is_secret_key(key) is secret


def test_redact_config_removes_every_secret_from_a_pydantic_model():
    snap = redact_config(_cfg())
    text = json.dumps(snap)
    assert "sk-live-abc" not in text
    assert "hunter2" not in text
    assert snap["openai_api_key"] == REDACTED
    assert snap["db"]["password"] == REDACTED
    assert snap["db"]["url"] == f"postgresql://elevate:{REDACTED}@localhost/db"
    assert snap["max_tokens"] == 512
    assert snap["paths"] == ["data/raw"]


def test_redact_config_handles_plain_mappings_and_nested_secret_sections():
    snap = redact_config(
        {"neo4j": {"auth_token": "t0k"}, "secrets": {"a": "x", "b": ["y"]}, "token_budget": 9}
    )
    assert snap == {
        "neo4j": {"auth_token": REDACTED},
        "secrets": {"a": REDACTED, "b": [REDACTED]},
        "token_budget": 9,
    }


def test_redact_config_rejects_other_types():
    with pytest.raises(TypeError):
        redact_config(["not", "a", "config"])  # type: ignore[arg-type]


# ---------------------------------------------------------------- environment capture


def test_git_commit_hash_degrades_outside_a_repository(tmp_path):
    assert git_commit_hash(tmp_path) == UNKNOWN_COMMIT


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_commit_hash_handles_empty_and_committed_repos(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    assert git_commit_hash(tmp_path) == UNKNOWN_COMMIT  # no commits yet
    (tmp_path / "f.txt").write_text("a")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    commit = git_commit_hash(tmp_path)
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    (tmp_path / "f.txt").write_text("b")
    assert git_commit_hash(tmp_path) == f"{commit}-dirty"


def test_library_versions_include_python_and_core_dependencies():
    versions = library_versions()
    assert {"python", "numpy", "pydantic"} <= versions.keys()
    assert "definitely-not-installed-pkg" not in library_versions(("definitely-not-installed-pkg",))


# ---------------------------------------------------------------- open_run


def test_open_run_captures_snapshot_seeds_versions_git_and_start_time(tmp_path):
    start = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    run = open_run(_cfg(), split="dev", pipeline="V1", repo_dir=tmp_path, started_at=start)

    assert run.dataset == "hotpotqa" and run.split == "dev" and run.pipeline == "V1"
    assert run.seeds.seed == 11  # taken from the config
    assert run.started_at == start
    assert run.git_commit == UNKNOWN_COMMIT
    assert "numpy" in run.library_versions
    assert run.config_snapshot["openai_api_key"] == REDACTED
    assert run.items == [] and run.summary is None
    assert run.run_id.startswith("run-V1-hotpotqa-dev-")


def test_open_run_never_serialises_secrets():
    dumped = open_run(_cfg()).model_dump_json()
    assert "sk-live-abc" not in dumped and "hunter2" not in dumped


def test_open_run_seed_precedence_and_default():
    assert open_run({"seed": 5}, seed=9).seeds.seed == 9
    assert open_run({}).seeds.seed == DEFAULT_SEED
    assert open_run({}).dataset == "unspecified"


def test_open_run_config_hash_is_stable_for_equal_configs():
    assert open_run(_cfg()).config_hash == open_run(_cfg()).config_hash
    assert open_run({"a": 1}).config_hash != open_run({"a": 2}).config_hash


def test_run_record_round_trips_through_json():
    run = open_run({"dataset": "musique"}, split="test", pipeline="mix")
    run.add_item(
        RunItem(
            question_id="q1",
            prediction="Paris",
            retrieved_chunk_ids=["chunk-1"],
            cited_chunk_ids=["chunk-1"],
            invalid_citations=[3],
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=12.5,
            cost_usd=0.001,
        )
    )
    restored = RunRecord.model_validate_json(run.model_dump_json())
    assert restored == run
    assert isinstance(restored.seeds, SeedRecord)


def test_run_item_rejects_negative_usage():
    with pytest.raises(ValueError):
        RunItem(question_id="q", prediction="", prompt_tokens=-1)
