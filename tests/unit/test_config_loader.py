"""Unit tests for the Config_Loader (Requirements 1.2, 1.3, 1.4, 3.8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from elevate.config import ConfigLoader, ExperimentConfig
from elevate.config.loader import DEFAULTS_PATH
from elevate.config.schema import ExperimentDefaults
from elevate.core.errors import ConfigError

SECRET = "sk-super-secret-value-123"


def _write(tmp_path: Path, text: str, name: str = "exp.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _loader(**kwargs) -> ConfigLoader:
    kwargs.setdefault("env_file", None)
    kwargs.setdefault("environ", {})
    return ConfigLoader(**kwargs)


def test_defaults_yaml_matches_schema_defaults(tmp_path):
    assert DEFAULTS_PATH.is_file()
    config = _loader().load(_write(tmp_path, "name: x\n"))
    shared = config.model_dump(exclude={"name", "description"})
    assert ExperimentDefaults.model_validate(shared) == ExperimentDefaults()


def test_load_returns_validated_config(tmp_path):
    path = _write(
        tmp_path,
        "name: e1\nseed: 7\nthresholds: {merge_threshold: 0.94}\nbackends: {vector: qdrant}\n",
    )
    config = _loader().load(path)
    assert isinstance(config, ExperimentConfig)
    assert config.seed == 7
    assert config.thresholds.merge_threshold == 0.94
    assert config.thresholds.rrf_k == 60  # untouched default kept
    assert config.backends.vector == "qdrant"
    assert config.caps.context_token_cap == 8000


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("name: e1\nbogus: 1\n", "bogus", "unknown field"),
        ("name: e1\nthresholds: {rrf_kk: 3}\n", "thresholds.rrf_kk", "unknown field"),
        ("seed: 1\n", "name", "str"),
        ("name: e1\nseed: abc\n", "seed", "int"),
        ("name: e1\nbackends: {graph: oracle}\n", "backends.graph", "one of 'networkx'"),
        ("name: e1\ncaps: {context_token_cap: [1]}\n", "caps.context_token_cap", "int"),
    ],
)
def test_invalid_field_names_file_field_and_type(tmp_path, text, field, expected):
    path = _write(tmp_path, text)
    with pytest.raises(ConfigError) as info:
        _loader().load(path)
    err = info.value
    assert err.file == str(path)
    assert err.field == field
    assert expected in err.expected
    assert str(path) in str(err) and field in str(err)


def test_error_in_defaults_file_names_defaults_file(tmp_path):
    defaults = _write(tmp_path, "seed: nope\n", "defaults.yaml")
    with pytest.raises(ConfigError) as info:
        _loader(defaults_path=defaults).load(_write(tmp_path, "name: e1\n"))
    assert info.value.file == str(defaults)
    assert info.value.field == "seed"


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="<file>"):
        _loader().load(tmp_path / "absent.yaml")
    with pytest.raises(ConfigError, match="valid YAML"):
        _loader().load(_write(tmp_path, "name: [unclosed\n"))
    with pytest.raises(ConfigError, match="mapping at the top level"):
        _loader().load(_write(tmp_path, "- a\n- b\n"))


def test_secrets_load_from_env_file_and_environment(tmp_path):
    env_file = _write(tmp_path, f"OPENAI_API_KEY={SECRET}\nNEO4J_PASSWORD=from-file\n", ".env")
    loader = _loader(env_file=env_file, environ={"NEO4J_PASSWORD": "from-env"})
    config = loader.load(_write(tmp_path, "name: e1\n"))
    assert isinstance(config.secrets.openai_api_key, SecretStr)
    assert config.secrets.openai_api_key.get_secret_value() == SECRET
    assert config.secrets.neo4j_password is not None
    assert config.secrets.neo4j_password.get_secret_value() == "from-env"  # environ wins
    # Secrets never appear in reprs or config snapshots.
    assert SECRET not in repr(config)
    assert SECRET not in str(config.secrets)
    assert "secrets" not in config.model_dump()
    assert SECRET not in config.model_dump_json()


def test_secrets_rejected_in_config_file(tmp_path):
    path = _write(tmp_path, f"name: e1\nsecrets: {{openai_api_key: {SECRET}}}\n")
    with pytest.raises(ConfigError) as info:
        _loader().load(path)
    assert info.value.field == "secrets"
    assert SECRET not in str(info.value)


def test_env_overrides_and_env_errors_hide_value(tmp_path):
    path = _write(tmp_path, "name: e1\nseed: 1\n")
    config = _loader(environ={"ELEVATE_SEED": "9", "ELEVATE_DATA_DIR": "/d"}).load(path)
    assert config.seed == 9
    assert config.paths.data_dir == Path("/d")
    with pytest.raises(ConfigError) as info:
        _loader(environ={"ELEVATE_RUN_SPENDING_CAP_USD": SECRET}).load(path)
    assert info.value.field == "ELEVATE_RUN_SPENDING_CAP_USD"
    assert info.value.expected == "float"
    assert SECRET not in str(info.value)


def test_judge_same_family_as_generator_rejected(tmp_path):
    path = _write(tmp_path, "name: e1\nmodels: {generator: gpt-gen, judge: gpt-judge}\n")
    families = {"gpt-gen": "openai", "gpt-judge": "openai"}
    with pytest.raises(ConfigError) as info:
        _loader(families=families).load(path)
    message = str(info.value)
    assert info.value.field == "models.judge"
    assert "gpt-gen" in message and "gpt-judge" in message


def test_judge_different_family_and_inline_family_accepted(tmp_path):
    path = _write(
        tmp_path,
        "name: e1\nmodels:\n  generator: qwen\n  judge: {key: judge-api, family: openai}\n",
    )
    config = _loader(families={"qwen": "qwen", "judge-api": "qwen"}).load(path)
    assert config.models.judge.key == "judge-api"
    assert config.models.extraction.key == "fake-llm"  # role default kept


def test_overridden_role_does_not_inherit_default_family(tmp_path):
    path = _write(tmp_path, "name: e1\nmodels: {generator: unknown-model}\n")
    with pytest.raises(ConfigError) as info:
        _loader().load(path)
    assert info.value.field == "models.generator.family"
    assert "unknown-model" in str(info.value)


def test_fake_models_exempt_from_family_rule(tmp_path):
    config = _loader().load(_write(tmp_path, "name: e1\n"))
    assert config.models.generator.key == config.models.judge.key == "fake-llm"
