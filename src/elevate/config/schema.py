"""Typed configuration schema (Requirement 1.2).

Every model forbids unknown fields (``extra="forbid"``) and is frozen, so a validated
configuration cannot drift after loading. Defaults follow the design's "Chosen defaults for
open thresholds" table and are repeated in ``configs/defaults.yaml``.

Secret values (API keys, database passwords, DSNs) never come from config files. They are
read from the environment or ``.env`` by the Config_Loader and stored as :class:`SecretStr`,
which masks the value in ``repr``/``str`` and is excluded from :meth:`model_dump`, so it
cannot reach logs, Run_Records or error messages (Requirement 1.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

FAKE_FAMILY = "fake"
"""Family of the deterministic offline fakes. Exempt from the judge/generator family rule
because fakes are scripted, not models with shared training data (Requirement 3.8)."""

VectorBackend = Literal["faiss", "nanovectordb", "qdrant", "milvus", "pgvector"]
GraphBackend = Literal["networkx", "neo4j", "memgraph", "postgres"]


class StrictModel(BaseModel):
    """Base for every config model: unknown fields are errors and values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRef(StrictModel):
    """A model role assignment: a Model_Registry key plus an optional inline family.

    The family normally comes from the registry entry; ``family`` may be given inline for
    entries the registry does not cover (e.g. the offline fakes). A bare string in YAML is
    shorthand for ``{key: <string>}``.
    """

    key: str = Field(min_length=1)
    family: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"key": data}
        return data


def _fake(key: str) -> ModelRef:
    return ModelRef(key=key, family=FAKE_FAMILY)


class ModelRoles(StrictModel):
    """Model assignment per role (Requirements 3.1, 3.7). Defaults are the offline fakes."""

    embedder: ModelRef = Field(default_factory=lambda: _fake("fake-embedder"))
    reranker: ModelRef = Field(default_factory=lambda: _fake("fake-reranker"))
    extraction: ModelRef = Field(default_factory=lambda: _fake("fake-llm"))
    generator: ModelRef = Field(default_factory=lambda: _fake("fake-llm"))
    judge: ModelRef = Field(default_factory=lambda: _fake("fake-llm"))


class BackendSettings(StrictModel):
    """Storage backends. Defaults are the no-setup backends (Requirement 2.3)."""

    vector: VectorBackend = "faiss"
    graph: GraphBackend = "networkx"


class ThresholdSettings(StrictModel):
    """Open thresholds with the design's chosen defaults; tuned on dev only."""

    contamination_em: float = Field(default=0.25, ge=0.0, le=1.0)  # Req 14.4
    merge_threshold: float = Field(default=0.88, ge=0.0, le=1.0)  # Req 16.1, 28.2
    entity_description_merge_limit: int = Field(default=8, ge=1)  # Req 16.4
    answer_entity_coverage_gate: float = Field(default=0.60, ge=0.0, le=1.0)  # Req 18.3
    rrf_k: int = Field(default=60, ge=1)  # Req 12.3
    reranker_top_n: int = Field(default=8, ge=5, le=10)  # Req 12.6
    judge_kappa_pass: float = Field(default=0.6, ge=-1.0, le=1.0)  # Req 10.5


class CapSettings(StrictModel):
    """Hard limits: context size and spending caps in USD (0 disables paid calls)."""

    context_token_cap: int = Field(default=8000, ge=1)  # Req 13, 28.3
    run_spending_cap_usd: float = Field(default=5.0, ge=0.0)  # Req 4.4
    project_spending_cap_usd: float = Field(default=200.0, ge=0.0)  # Req 4.4


class RetrySettings(StrictModel):
    """Transient-error policy for the LLM_Gateway (Requirement 3.5)."""

    max_attempts: int = Field(default=5, ge=1)
    backoff_base_s: float = Field(default=2.0, gt=0.0)


class PathSettings(StrictModel):
    """Working directories; overridable with the ``ELEVATE_*_DIR`` environment variables."""

    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    cache_dir: Path = Path("./.cache/elevate")
    trace_dir: Path = Path("./logs/traces")


class Endpoints(StrictModel):
    """Non-secret connection settings read from the environment or ``.env``."""

    openai_base_url: str | None = None
    ollama_base_url: str | None = None
    vllm_base_url: str | None = None
    hf_home: str | None = None
    langfuse_host: str | None = None
    qdrant_url: str | None = None
    milvus_uri: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    memgraph_uri: str | None = None
    memgraph_user: str | None = None


class Secrets(StrictModel):
    """API keys and database credentials, only ever read from the environment or ``.env``.

    DSNs are secret because they embed a password.
    """

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    vllm_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    qdrant_api_key: SecretStr | None = None
    milvus_token: SecretStr | None = None
    pgvector_dsn: SecretStr | None = None
    neo4j_password: SecretStr | None = None
    memgraph_password: SecretStr | None = None
    postgres_graph_dsn: SecretStr | None = None


class ExperimentDefaults(StrictModel):
    """Settings shared by every experiment; the schema of ``configs/defaults.yaml``."""

    seed: int = 42
    backends: BackendSettings = Field(default_factory=BackendSettings)
    models: ModelRoles = Field(default_factory=ModelRoles)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    caps: CapSettings = Field(default_factory=CapSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class ExperimentConfig(ExperimentDefaults):
    """A validated experiment configuration (Requirement 1.2).

    ``endpoints`` and ``secrets`` are filled from the environment by the Config_Loader, never
    from the file, and are excluded from :meth:`model_dump` so a config snapshot written to a
    Run_Record cannot contain them.
    """

    name: str = Field(min_length=1)
    description: str = ""
    endpoints: Endpoints = Field(default_factory=Endpoints, exclude=True, repr=False)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True, repr=False)


ENV_ONLY_SECTIONS = ("endpoints", "secrets")
"""Top-level ExperimentConfig fields that config files may not set."""

__all__ = [
    "ENV_ONLY_SECTIONS",
    "FAKE_FAMILY",
    "BackendSettings",
    "CapSettings",
    "Endpoints",
    "ExperimentConfig",
    "ExperimentDefaults",
    "GraphBackend",
    "ModelRef",
    "ModelRoles",
    "PathSettings",
    "RetrySettings",
    "Secrets",
    "StrictModel",
    "ThresholdSettings",
    "VectorBackend",
]
