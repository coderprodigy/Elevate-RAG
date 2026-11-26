"""Config_Loader: YAML experiment configs + environment -> validated ExperimentConfig.

Load order, later sources overriding earlier ones:

1. schema defaults (:mod:`elevate.config.schema`);
2. ``configs/defaults.yaml`` (validated on its own, so its errors name that file);
3. the experiment file;
4. ``ELEVATE_*`` variables from ``.env``, then from the process environment.

Model roles (``models.<role>``) are replaced as a whole when overridden, never merged field
by field, so an inline ``family`` from the defaults cannot leak onto a different model.

Any invalid input stops loading with a :class:`ConfigError` naming the file, the field and the
expected type (Requirement 1.3). Error messages are built from the field location and the
schema only, never from the offending value, so secrets cannot leak through them
(Requirement 1.4). A judge and generator in the same model family are rejected
(Requirement 3.8).
"""

from __future__ import annotations

import os
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from elevate.config.schema import (
    ENV_ONLY_SECTIONS,
    FAKE_FAMILY,
    Endpoints,
    ExperimentConfig,
    ExperimentDefaults,
    ModelRef,
    Secrets,
)
from elevate.core.errors import ConfigError

DEFAULTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "defaults.yaml"
ENV_FILE = Path(".env")
ENVIRONMENT_SOURCE = "<environment>"

ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "ELEVATE_SEED": ("seed",),
    "ELEVATE_RUN_SPENDING_CAP_USD": ("caps", "run_spending_cap_usd"),
    "ELEVATE_PROJECT_SPENDING_CAP_USD": ("caps", "project_spending_cap_usd"),
    "ELEVATE_DATA_DIR": ("paths", "data_dir"),
    "ELEVATE_RESULTS_DIR": ("paths", "results_dir"),
    "ELEVATE_CACHE_DIR": ("paths", "cache_dir"),
    "ELEVATE_TRACE_DIR": ("paths", "trace_dir"),
}
"""Environment variables that override config-file settings, mapped to their field path."""

# Endpoint and secret fields are named after their environment variable, lower-cased.
ENDPOINT_VARS: dict[str, str] = {name.upper(): name for name in Endpoints.model_fields}
SECRET_VARS: dict[str, str] = {name.upper(): name for name in Secrets.model_fields}

_EnvSource = tuple[str, str]  # (variable name, source label)
_M = TypeVar("_M", bound=BaseModel)


class ConfigLoader:
    """Loads and validates experiment configs (Requirements 1.2, 1.3, 1.4, 3.8).

    Args:
        defaults_path: shared defaults file; ``None`` skips it.
        env_file: ``.env`` file to read if it exists; ``None`` skips it.
        environ: process environment (defaults to :data:`os.environ`); wins over ``env_file``.
        families: Model_Registry key -> model family, used for the judge/generator check.
            An inline ``family`` on a model role takes precedence.
    """

    def __init__(
        self,
        *,
        defaults_path: Path | None = DEFAULTS_PATH,
        env_file: Path | None = ENV_FILE,
        environ: Mapping[str, str] | None = None,
        families: Mapping[str, str] | None = None,
    ) -> None:
        self.defaults_path = defaults_path
        self.env_file = env_file
        self.environ = os.environ if environ is None else environ
        self.families = dict(families or {})

    def load(self, path: Path) -> ExperimentConfig:
        """Validate ``path`` and return the configuration, or raise :class:`ConfigError`."""
        path = Path(path)
        merged: dict[str, Any] = {}
        if self.defaults_path is not None:
            raw_defaults = _read_yaml(self.defaults_path)
            _reject_env_only(raw_defaults, self.defaults_path)
            defaults = _validate(ExperimentDefaults, raw_defaults, str(self.defaults_path), {})
            merged = defaults.model_dump(mode="json")

        raw = _read_yaml(path)
        _reject_env_only(raw, path)
        merged = _merge(merged, raw)

        env, env_sources = self._read_env()
        env_locs: dict[tuple[str, ...], _EnvSource] = {}
        for var, loc in ENV_OVERRIDES.items():
            if var in env:
                _set_path(merged, loc, env[var])
                env_locs[loc] = (var, env_sources[var])
        merged["endpoints"] = {f: env[v] for v, f in ENDPOINT_VARS.items() if v in env}
        merged["secrets"] = {f: env[v] for v, f in SECRET_VARS.items() if v in env}

        config = _validate(ExperimentConfig, merged, str(path), env_locs)
        check_judge_family(config, self.families, path)
        return config

    def _read_env(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return non-empty values of known variables and the source each came from."""
        known = ENV_OVERRIDES.keys() | ENDPOINT_VARS.keys() | SECRET_VARS.keys()
        values: dict[str, str] = {}
        sources: dict[str, str] = {}
        if self.env_file is not None and self.env_file.is_file():
            for var, value in dotenv_values(self.env_file).items():
                if var in known and value:
                    values[var] = value
                    sources[var] = str(self.env_file)
        for var in known:
            value = self.environ.get(var)
            if value:
                values[var] = value
                sources[var] = ENVIRONMENT_SOURCE
        return values, sources


def load_config(path: Path, **kwargs: Any) -> ExperimentConfig:
    """Convenience wrapper: ``ConfigLoader(**kwargs).load(path)``."""
    return ConfigLoader(**kwargs).load(path)


def resolve_family(ref: ModelRef, families: Mapping[str, str]) -> str | None:
    """Inline family first, then the registry mapping; ``None`` if unknown."""
    return ref.family or families.get(ref.key)


def check_judge_family(
    config: ExperimentConfig, families: Mapping[str, str], file: str | Path
) -> None:
    """Reject a judge from the generator's model family, naming both models (Req 3.8)."""
    judge, generator = config.models.judge, config.models.generator
    for role, ref in (("judge", judge), ("generator", generator)):
        if resolve_family(ref, families) is None:
            raise ConfigError(
                file=file,
                field=f"models.{role}.family",
                expected="str",
                detail=f"no model family known for '{ref.key}'; add it to the Model_Registry "
                "or set 'family' inline",
            )
    judge_family = resolve_family(judge, families)
    if judge_family == resolve_family(generator, families) and judge_family != FAKE_FAMILY:
        raise ConfigError(
            file=file,
            field="models.judge",
            expected="a model from a different family than the generator",
            detail=f"judge '{judge.key}' and generator '{generator.key}' are both in "
            f"family '{judge_family}'",
        )


# ---------------------------------------------------------------- helpers


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(file=path, field="<file>", expected="an existing YAML file")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else None
        # The YAML error text can quote file content, so only the position is reported.
        raise ConfigError(
            file=path, field="<document>", expected="valid YAML", detail=where
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            file=path, field="<file>", expected="a readable UTF-8 file", detail=type(exc).__name__
        ) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(file=path, field="<document>", expected="a mapping at the top level")
    return data


def _reject_env_only(raw: Mapping[str, Any], path: Path) -> None:
    for section in ENV_ONLY_SECTIONS:
        if section in raw:
            raise ConfigError(
                file=path,
                field=section,
                expected="no such field in config files",
                detail=f"'{section}' are read only from environment variables or .env",
            )


def _merge(
    base: Mapping[str, Any], override: Mapping[str, Any], at: tuple[str, ...] = ()
) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        nested = isinstance(value, dict) and isinstance(out.get(key), dict)
        if nested and at != ("models",):
            out[key] = _merge(out[key], value, (*at, str(key)))
        else:
            out[key] = value
    return out


def _set_path(data: dict[str, Any], loc: tuple[str, ...], value: Any) -> None:
    node = data
    for part in loc[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[loc[-1]] = value


def _validate(
    model: type[_M],
    data: Mapping[str, Any],
    file: str,
    env_locs: Mapping[tuple[str, ...], _EnvSource],
) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise _to_config_error(exc, model, file, env_locs) from None


def _to_config_error(
    exc: ValidationError,
    model: type[BaseModel],
    file: str,
    env_locs: Mapping[tuple[str, ...], _EnvSource],
) -> ConfigError:
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    err = errors[0]
    loc = tuple(str(p) for p in err["loc"])
    field = ".".join(loc) or "<document>"
    more = f"; {len(errors) - 1} more error(s)" if len(errors) > 1 else ""

    if err["type"] == "extra_forbidden":
        parent = _annotation_at(model, err["loc"][:-1])
        allowed = (
            ", ".join(parent.model_fields)
            if isinstance(parent, type) and issubclass(parent, BaseModel)
            else "see schema"
        )
        return ConfigError(
            file=file,
            field=field,
            expected="no such field (unknown field)",
            detail=f"allowed fields: {allowed}{more}",
        )

    expected_tp = _annotation_at(model, err["loc"])
    expected = _type_name(expected_tp) if expected_tp is not None else "see schema"
    detail = f"{err['type']}: {err['msg']}{more}"
    for env_loc, (var, source) in env_locs.items():
        if loc[: len(env_loc)] == env_loc:
            return ConfigError(file=source, field=var, expected=expected, detail=detail)
    return ConfigError(file=file, field=field, expected=expected, detail=detail)


def _annotation_at(model: type[BaseModel], loc: tuple[int | str, ...]) -> Any:
    """Return the schema annotation at ``loc``, or ``None`` if it cannot be resolved."""
    tp: Any = model
    for part in loc:
        tp = _strip_optional(tp)
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            if not isinstance(part, str) or part not in tp.model_fields:
                return None
            tp = tp.model_fields[part].annotation
            continue
        origin = get_origin(tp)
        args = get_args(tp)
        if origin in (list, tuple, set, frozenset) and isinstance(part, int) and args:
            tp = args[0]
        elif origin is dict and len(args) == 2:
            tp = args[1]
        else:
            return None
    return tp


def _strip_optional(tp: Any) -> Any:
    if get_origin(tp) in (Union, types.UnionType):
        members = [a for a in get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return tp


def _type_name(tp: Any) -> str:
    if tp is ModelRef:
        return "a Model_Registry key (str) or a mapping with fields key, family"
    origin = get_origin(tp)
    if origin is Literal:
        return "one of " + ", ".join(repr(a) for a in get_args(tp))
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        names = [_type_name(a) for a in args if a is not type(None)]
        if type(None) in args:
            names.append("null")
        return " or ".join(names)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return f"a mapping with fields {', '.join(tp.model_fields)}"
    if isinstance(tp, type) and not get_args(tp):
        return tp.__name__
    return str(tp).replace("typing.", "")


__all__ = [
    "DEFAULTS_PATH",
    "ENDPOINT_VARS",
    "ENVIRONMENT_SOURCE",
    "ENV_OVERRIDES",
    "SECRET_VARS",
    "ConfigLoader",
    "check_judge_family",
    "load_config",
    "resolve_family",
]
