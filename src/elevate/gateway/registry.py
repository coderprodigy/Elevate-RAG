"""Model_Registry: the frozen list of model entries per role (Requirement 3.1).

``configs/models.yaml`` holds every model the project may call. Each entry records the
provider, exact model ID, version, model family, context length and price per million input
and output tokens. Profiles map each role to an entry key; experiment configs may override a
role with another key. Entries still marked ``choose_at_freeze`` are listed but cannot be
resolved for a Run.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from elevate.core.errors import ConfigError

DEFAULT_REGISTRY_PATH = Path("configs/models.yaml")

Provider = Literal[
    "fake", "sentence-transformers", "openai", "anthropic", "google", "ollama", "vllm"
]
Status = Literal["verified", "choose_at_freeze"]

# The judge/generator family rule (Requirement 3.8) applies to real models only; the offline
# fake plays every role.
FAKE_FAMILY = "fake"


class ModelRole(StrEnum):
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    EXTRACTION = "extraction"
    GENERATOR = "generator"
    JUDGE = "judge"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _EntryBase(_Strict):
    roles: tuple[ModelRole, ...] = Field(min_length=1)
    provider: Provider
    family: str = Field(min_length=1)
    supports_temperature: bool = True
    supports_seed: bool
    base_url_env: str | None = None
    api_key_env: str | None = None
    thinking: Literal["on", "off"] | None = None
    notes: str | None = None


class ModelEntry(_EntryBase):
    """A frozen, resolvable registry entry."""

    key: str
    status: Literal["verified"] = "verified"
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    context_length: PositiveInt
    price_in_per_mtok: NonNegativeFloat
    price_out_per_mtok: NonNegativeFloat

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Dollar cost of one call at this entry's frozen prices."""
        return (
            prompt_tokens * self.price_in_per_mtok + completion_tokens * self.price_out_per_mtok
        ) / 1_000_000


class PendingEntry(_EntryBase):
    """An entry whose exact model is chosen on the freeze date; fields may be null."""

    key: str
    status: Literal["choose_at_freeze"]
    model_id: str | None = None
    version: str | None = None
    context_length: PositiveInt | None = None
    price_in_per_mtok: NonNegativeFloat | None = None
    price_out_per_mtok: NonNegativeFloat | None = None


class _RegistryFile(_Strict):
    schema_version: Literal[1]
    default_profile: str
    profiles: dict[str, dict[ModelRole, str]]
    models: dict[str, dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_profile_exists(self) -> _RegistryFile:
        if self.default_profile not in self.profiles:
            raise ValueError(f"default_profile '{self.default_profile}' is not in profiles")
        return self


def _first_error(exc: ValidationError, prefix: str) -> tuple[str, str, str]:
    err = exc.errors()[0]
    loc = ".".join(str(p) for p in (*prefix.split("."), *err["loc"]) if p != "")
    return loc or "<root>", err["type"], err["msg"]


class ModelRegistry:
    """Lookup over frozen and pending entries, with role profiles."""

    def __init__(
        self,
        *,
        entries: Mapping[str, ModelEntry],
        pending: Mapping[str, PendingEntry],
        profiles: Mapping[str, Mapping[ModelRole, str]],
        default_profile: str,
        source: str | Path = "<memory>",
    ) -> None:
        self._entries = dict(entries)
        self._pending = dict(pending)
        self._profiles = {name: dict(roles) for name, roles in profiles.items()}
        self.default_profile = default_profile
        self.source = str(source)
        self._validate_profiles()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
        """Load and validate a registry file; raise :class:`ConfigError` on any problem."""
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ConfigError(
                file=path, field="<file>", expected="an existing YAML file", detail="not found"
            ) from None
        except yaml.YAMLError as exc:
            raise ConfigError(
                file=path, field="<file>", expected="valid YAML", detail=str(exc)
            ) from None
        if not isinstance(raw, dict):
            raise ConfigError(file=path, field="<root>", expected="a mapping")
        return cls.from_mapping(raw, source=path)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str | Path) -> ModelRegistry:
        try:
            parsed = _RegistryFile.model_validate(raw)
        except ValidationError as exc:
            field, expected, msg = _first_error(exc, "")
            raise ConfigError(file=source, field=field, expected=expected, detail=msg) from None

        entries: dict[str, ModelEntry] = {}
        pending: dict[str, PendingEntry] = {}
        for key, body in parsed.models.items():
            if not isinstance(body, dict):
                raise ConfigError(file=source, field=f"models.{key}", expected="a mapping")
            data = {**body, "key": key}
            status = body.get("status", "verified")
            model: type[ModelEntry] | type[PendingEntry] = (
                PendingEntry if status == "choose_at_freeze" else ModelEntry
            )
            try:
                validated = model.model_validate(data)
            except ValidationError as exc:
                field, expected, msg = _first_error(exc, f"models.{key}")
                raise ConfigError(file=source, field=field, expected=expected, detail=msg) from None
            if isinstance(validated, ModelEntry):
                entries[key] = validated
            else:
                pending[key] = validated
        return cls(
            entries=entries,
            pending=pending,
            profiles=parsed.profiles,
            default_profile=parsed.default_profile,
            source=source,
        )

    def _validate_profiles(self) -> None:
        for name, roles in self._profiles.items():
            for role in ModelRole:
                field = f"profiles.{name}.{role.value}"
                key = roles.get(role)
                if key is None:
                    raise ConfigError(file=self.source, field=field, expected="an entry key")
                self._check_key_role(key, role, field)

    def _known(self, key: str, field: str) -> ModelEntry | PendingEntry:
        known = self._entries.get(key) or self._pending.get(key)
        if known is None:
            raise ConfigError(
                file=self.source,
                field=field,
                expected="a key under models",
                detail=f"unknown model '{key}'",
            )
        return known

    def _check_key_role(self, key: str, role: ModelRole, field: str) -> None:
        known = self._known(key, field)
        if role not in known.roles:
            raise ConfigError(
                file=self.source,
                field=field,
                expected=f"an entry with role '{role.value}'",
                detail=f"'{key}' has roles {[r.value for r in known.roles]}",
            )

    # ------------------------------------------------------------------ lookup
    def __contains__(self, key: object) -> bool:
        return key in self._entries or key in self._pending

    def __iter__(self) -> Iterator[str]:
        yield from self._entries
        yield from self._pending

    @property
    def profiles(self) -> dict[str, dict[ModelRole, str]]:
        return {name: dict(roles) for name, roles in self._profiles.items()}

    def pending(self) -> dict[str, PendingEntry]:
        return dict(self._pending)

    def get(self, key: str) -> ModelEntry:
        """Return the frozen entry ``key``; pending or unknown keys raise ConfigError."""
        if key in self._entries:
            return self._entries[key]
        if key in self._pending:
            raise ConfigError(
                file=self.source,
                field=f"models.{key}",
                expected="status 'verified' with model_id, version, context_length and prices",
                detail="entry is still 'choose_at_freeze'; freeze it before use",
            )
        raise ConfigError(
            file=self.source,
            field=f"models.{key}",
            expected="a key under models",
            detail=f"unknown model '{key}'",
        )

    def entry(
        self, role: ModelRole | str, *, key: str | None = None, profile: str | None = None
    ) -> ModelEntry:
        """Resolve the entry for ``role``: an explicit ``key`` or the profile's assignment."""
        role = ModelRole(role)
        if key is None:
            name = profile or self.default_profile
            if name not in self._profiles:
                raise ConfigError(
                    file=self.source,
                    field="profiles",
                    expected=f"a profile named '{name}'",
                    detail=f"available: {sorted(self._profiles)}",
                )
            key = self._profiles[name][role]
            field = f"profiles.{name}.{role.value}"
        else:
            field = f"models.{key}"
        self._check_key_role(key, role, field)
        return self.get(key)

    def check_judge_generator_families(self, *, generator: str, judge: str) -> None:
        """Raise ConfigError naming both models if they share a family (Requirement 3.8).

        The offline fake family is exempt so one fake LLM can play every role.
        """
        gen = self._known(generator, f"models.{generator}")
        jud = self._known(judge, f"models.{judge}")
        if gen.family == jud.family and gen.family != FAKE_FAMILY:
            raise ConfigError(
                file=self.source,
                field="judge",
                expected="a judge from a different model family than the generator",
                detail=(
                    f"judge '{judge}' and generator '{generator}' are both family '{gen.family}'"
                ),
            )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "FAKE_FAMILY",
    "ModelEntry",
    "ModelRegistry",
    "ModelRole",
    "PendingEntry",
    "Provider",
    "Status",
]
