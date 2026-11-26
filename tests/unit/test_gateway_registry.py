from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from elevate.core.errors import ConfigError
from elevate.gateway.registry import ModelEntry, ModelRegistry, ModelRole

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = REPO_ROOT / "configs" / "models.yaml"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.load(REGISTRY_FILE)


@pytest.fixture
def raw() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


# ---------------------------------------------------------------- shipped registry


def test_every_role_resolves_offline_to_a_fake(registry: ModelRegistry) -> None:
    for role in ModelRole:
        entry = registry.entry(role)
        assert entry.provider == "fake"
        assert role in entry.roles


def test_entries_record_req_3_1_fields(registry: ModelRegistry) -> None:
    entry = registry.get("claude-sonnet-5.5")
    assert entry.provider == "anthropic"
    assert entry.model_id == "claude-sonnet-5-5"
    assert entry.version
    assert entry.family == "anthropic"
    assert entry.context_length == 1_000_000
    assert (entry.price_in_per_mtok, entry.price_out_per_mtok) == (2.0, 10.0)
    assert entry.supports_temperature is False
    assert entry.supports_seed is False


def test_registry_matches_models_doc(registry: ModelRegistry) -> None:
    expected_ids = {
        "fake-embedder": "fake-embedder",
        "fake-reranker": "fake-reranker",
        "fake-llm": "fake-llm",
        "bge-m3": "BAAI/bge-m3",
        "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
        "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        "ms-marco-minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
        "gemma-4-12b": "google/gemma-4-12B-it",
        "claude-sonnet-5.5": "claude-sonnet-5-5",
    }
    assert {k: registry.get(k).model_id for k in expected_ids} == expected_ids
    assert set(registry.pending()) == {"openai-embedding", "extractor-api", "judge-api"}
    assert set(registry) == set(expected_ids) | set(registry.pending())


def test_every_role_has_at_least_one_real_entry(registry: ModelRegistry) -> None:
    pending = registry.pending()
    entries = [pending[k] if k in pending else registry.get(k) for k in registry]
    for role in ModelRole:
        assert any(role in e.roles and e.provider != "fake" for e in entries)


def test_pending_entry_cannot_be_resolved(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigError) as info:
        registry.entry(ModelRole.JUDGE, profile="reference")
    assert info.value.field == "models.judge-api"
    assert "choose_at_freeze" in str(info.value)


def test_explicit_key_overrides_profile(registry: ModelRegistry) -> None:
    assert registry.entry("generator", key="gemma-4-12b").model_id == "google/gemma-4-12B-it"
    # The same Qwen entry serves extraction and generation (Requirement 3.7).
    assert registry.entry("extraction", key="qwen3.6-35b-a3b").key == "qwen3.6-35b-a3b"


def test_key_with_wrong_role_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigError, match="role 'judge'"):
        registry.entry(ModelRole.JUDGE, key="bge-m3")


def test_unknown_key_and_profile_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigError, match="unknown model 'nope'"):
        registry.get("nope")
    with pytest.raises(ConfigError, match="profile named 'nope'"):
        registry.entry("embedder", profile="nope")


def test_cost_uses_per_million_prices(registry: ModelRegistry) -> None:
    claude = registry.get("claude-sonnet-5.5")
    assert claude.cost_usd(1_000_000, 0) == pytest.approx(2.0)
    assert claude.cost_usd(500_000, 100_000) == pytest.approx(1.0 + 1.0)
    assert registry.get("fake-llm").cost_usd(10**6, 10**6) == 0.0


# ---------------------------------------------------------------- family rule helper


def test_same_family_judge_and_generator_rejected(registry: ModelRegistry) -> None:
    registry.check_judge_generator_families(generator="claude-sonnet-5.5", judge="judge-api")
    registry.check_judge_generator_families(generator="fake-llm", judge="fake-llm")
    with pytest.raises(ConfigError) as info:
        registry.check_judge_generator_families(
            generator="qwen3.6-35b-a3b", judge="qwen3.6-35b-a3b"
        )
    assert "qwen3.6-35b-a3b" in str(info.value)


def test_family_rule_names_both_models() -> None:
    reg = ModelRegistry.load(REGISTRY_FILE)
    with pytest.raises(ConfigError) as info:
        reg.check_judge_generator_families(generator="extractor-api", judge="judge-api")
    assert "extractor-api" in str(info.value)
    assert "judge-api" in str(info.value)


# ---------------------------------------------------------------- validation errors


def _load(tmp_path: Path, data: dict[str, Any]) -> ModelRegistry:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return ModelRegistry.load(path)


def test_missing_price_names_file_and_field(tmp_path: Path, raw: dict[str, Any]) -> None:
    del raw["models"]["bge-m3"]["price_in_per_mtok"]
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, raw)
    assert info.value.field == "models.bge-m3.price_in_per_mtok"
    assert info.value.file.endswith("models.yaml")


def test_wrong_type_and_unknown_field_rejected(tmp_path: Path, raw: dict[str, Any]) -> None:
    raw["models"]["bge-m3"]["context_length"] = "long"
    with pytest.raises(ConfigError, match=r"models\.bge-m3\.context_length"):
        _load(tmp_path, raw)
    raw["models"]["bge-m3"]["context_length"] = 8192
    raw["models"]["bge-m3"]["colour"] = "blue"
    with pytest.raises(ConfigError, match=r"models\.bge-m3\.colour"):
        _load(tmp_path, raw)


def test_negative_price_and_bad_provider_rejected(tmp_path: Path, raw: dict[str, Any]) -> None:
    raw["models"]["fake-llm"]["price_out_per_mtok"] = -1
    with pytest.raises(ConfigError, match="price_out_per_mtok"):
        _load(tmp_path, raw)
    raw["models"]["fake-llm"]["price_out_per_mtok"] = 0
    raw["models"]["fake-llm"]["provider"] = "acme"
    with pytest.raises(ConfigError, match="provider"):
        _load(tmp_path, raw)


def test_profile_must_cover_every_role_with_matching_entry(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    del raw["profiles"]["offline"]["judge"]
    with pytest.raises(ConfigError, match=r"profiles\.offline\.judge"):
        _load(tmp_path, raw)
    raw["profiles"]["offline"]["judge"] = "fake-embedder"
    with pytest.raises(ConfigError, match="role 'judge'"):
        _load(tmp_path, raw)


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        ModelRegistry.load(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="valid YAML"):
        ModelRegistry.load(bad)


def test_model_entry_is_immutable(registry: ModelRegistry) -> None:
    entry: ModelEntry = registry.get("bge-m3")
    with pytest.raises(ValidationError):
        entry.model_id = "other"  # type: ignore[misc]
