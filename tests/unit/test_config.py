from pathlib import Path

import pytest
from pydantic import ValidationError

from hgrag.config import (
    PROJECT_ROOT,
    ConfigError,
    ModelsConfig,
    deep_merge,
    load_config,
)


def test_project_config_loads_and_resolves_paths() -> None:
    config = load_config()
    assert config.paths.ledger_db.is_absolute()
    assert config.paths.ledger_db.is_relative_to(PROJECT_ROOT)
    assert "claude-haiku-5-5" in config.models.models


def test_claude_5_5_models_never_send_temperature() -> None:
    models = load_config().models.models
    for key in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        assert models[key].supports_temperature is False


def test_deep_merge_merges_nested_and_replaces_leaves() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    merged = deep_merge(base, {"a": {"y": 20, "z": 30}, "b": 2})
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 2}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}, "inputs must not be mutated"


def test_override_file_is_applied(tmp_path: Path) -> None:
    override = tmp_path / "exp.yaml"
    override.write_text("generation:\n  max_tokens: 77\n")
    assert load_config([override]).generation.max_tokens == 77


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    override = tmp_path / "typo.yaml"
    override.write_text("generation:\n  max_tokenz: 77\n")
    with pytest.raises(ValidationError):
        load_config([override])


def _models(roles: dict[str, str | None]) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {"fake": {"kind": "fake"}},
            "models": {
                "m": {
                    "provider": "fake",
                    "model": "m",
                    "price": {"input_per_mtok": 1, "output_per_mtok": 1},
                }
            },
            "roles": roles,
        }
    )


def test_role_pointing_to_unknown_model_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown model"):
        _models({"generator": "does-not-exist"})


def test_unset_role_gives_clear_error() -> None:
    models = _models({"judge": None})
    with pytest.raises(ConfigError, match="no model yet"):
        models.model_key_for_role("judge")
    with pytest.raises(ConfigError, match="unknown role"):
        models.model_key_for_role("nope")
