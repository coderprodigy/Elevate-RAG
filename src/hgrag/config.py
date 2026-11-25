"""Layered YAML configuration, validated with Pydantic.

Load order: ``configs/base.yaml`` (plus ``models``, ``budget``, ``datasets``, ``retrieval``),
then any override files given by the caller, merged key by key. Relative paths resolve against the
project root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Price(_Strict):
    """USD per million tokens. A missing cached price means cached tokens bill as normal input."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0)


class ProviderSpec(_Strict):
    kind: Literal["anthropic", "openai_compat", "fake"]
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    max_tokens_field: str = "max_tokens"
    max_retries: int = Field(default=3, ge=0)
    timeout_s: float = Field(default=120.0, gt=0)


class ModelSpec(_Strict):
    provider: str
    model: str
    price: Price
    supports_temperature: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(_Strict):
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    roles: dict[str, str | None]

    @model_validator(mode="after")
    def _check_references(self) -> ModelsConfig:
        for key, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"model '{key}' uses unknown provider '{spec.provider}'")
        for role, model_key in self.roles.items():
            if model_key is not None and model_key not in self.models:
                raise ValueError(f"role '{role}' points to unknown model '{model_key}'")
        return self

    def model_key_for_role(self, role: str) -> str:
        if role not in self.roles:
            raise ConfigError(f"unknown role '{role}'; known roles: {sorted(self.roles)}")
        model_key = self.roles[role]
        if model_key is None:
            raise ConfigError(f"role '{role}' has no model yet; set it in configs/models.yaml")
        return model_key


class BudgetConfig(_Strict):
    global_cap_usd: float = Field(gt=0)
    per_run_cap_usd: float = Field(gt=0)
    confirm_above_usd: float = Field(ge=0)


class PathsConfig(_Strict):
    data_dir: Path
    runs_dir: Path
    cache_db: Path
    ledger_db: Path

    def resolved(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path made absolute against ``root``."""
        values = {name: _absolute(getattr(self, name), root) for name in type(self).model_fields}
        return PathsConfig(**values)


class GenerationDefaults(_Strict):
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=1024, gt=0)


class DatasetSettings(_Strict):
    """How one dataset is sampled and split. See configs/datasets.yaml."""

    source_url: str
    license: str
    sample_size: int | None = Field(default=None, gt=0)  # None = use every question
    unanswerable_sample_size: int | None = Field(default=None, ge=0)
    max_distractors_per_question: int | None = Field(default=None, ge=0)
    dev_fraction: float = Field(default=0.2, gt=0, lt=1)


class ChunkerSpec(_Strict):
    kind: Literal["sentence", "paragraph"]
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)


class EmbedderSpec(_Strict):
    model: str  # Hugging Face model ID
    query_prefix: str = ""
    query_prompt_name: str | None = None  # use the model's own named prompt for queries
    batch_size: int = Field(default=32, gt=0)
    max_seq_length: int = Field(default=512, gt=0)


class RerankerSpec(_Strict):
    model: str
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=512, gt=0)


class HNSWSpec(_Strict):
    m: int = Field(gt=0)
    ef_construction: int = Field(gt=0)
    ef_search: int = Field(gt=0)


class SearchSettings(_Strict):
    candidates: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    rerank_candidates: int = Field(gt=0)
    final_k: int = Field(gt=0)
    context_token_budget: int = Field(gt=0)


class RetrievalConfig(_Strict):
    chunkers: dict[str, ChunkerSpec]
    dataset_chunker: dict[str, str]
    embedders: dict[str, EmbedderSpec]
    reference_embedder: str
    rerankers: dict[str, RerankerSpec]
    reference_reranker: str
    hnsw: HNSWSpec
    search: SearchSettings

    @model_validator(mode="after")
    def _check_references(self) -> RetrievalConfig:
        unknown = set(self.dataset_chunker.values()) - set(self.chunkers)
        if unknown:
            raise ValueError(f"dataset_chunker uses unknown chunkers {sorted(unknown)}")
        if self.reference_embedder not in self.embedders:
            raise ValueError(f"unknown reference_embedder '{self.reference_embedder}'")
        if self.reference_reranker not in self.rerankers:
            raise ValueError(f"unknown reference_reranker '{self.reference_reranker}'")
        return self


class GraphBackend(_Strict):
    kv_storage: str
    vector_storage: str
    graph_storage: str
    doc_status_storage: str


class GraphExtraction(_Strict):
    max_gleaning: int = Field(ge=0)
    llm_max_async: int = Field(gt=0)
    embedding_max_async: int = Field(gt=0)
    sample_fraction: float = Field(gt=0, le=1)
    est_input_tokens_per_chunk: int = Field(gt=0)
    est_output_tokens_per_chunk: int = Field(gt=0)


class GraphQuery(_Strict):
    top_k: int = Field(gt=0)
    chunk_top_k: int = Field(gt=0)
    entity_token_share: float = Field(gt=0, lt=1)
    relation_token_share: float = Field(gt=0, lt=1)
    rerank_chunks: bool
    est_keyword_input_tokens: int = Field(gt=0)
    est_keyword_output_tokens: int = Field(gt=0)


class GraphConfig(_Strict):
    reference_backend: str
    backends: dict[str, GraphBackend]
    extraction: GraphExtraction
    query: GraphQuery

    @model_validator(mode="after")
    def _check_backend(self) -> GraphConfig:
        if self.reference_backend not in self.backends:
            raise ValueError(f"unknown reference_backend '{self.reference_backend}'")
        return self


class RouterRules(_Strict):
    global_patterns: list[str]
    comparison_patterns: list[str]
    temporal_patterns: list[str]
    multi_hop_min_entities: int = Field(gt=0)


class RouterClassifier(_Strict):
    max_features: int = Field(gt=0)
    ngram_range: tuple[int, int]
    c: float = Field(gt=0)
    cv_folds: int = Field(ge=2)


class RouterFallback(_Strict):
    cheap_path: str
    escalate_path: str
    thresholds: list[float]


class RouterConfig(_Strict):
    paths: dict[str, str]  # path name → pipeline
    baseline_path: str
    type_to_path: dict[str, str]
    rules: RouterRules
    classifier: RouterClassifier
    fallback: RouterFallback

    @model_validator(mode="after")
    def _check_paths(self) -> RouterConfig:
        known = set(self.paths)
        used = {self.baseline_path, self.fallback.cheap_path, self.fallback.escalate_path}
        unknown = (used | set(self.type_to_path.values())) - known
        if unknown:
            raise ValueError(f"router refers to unknown paths {sorted(unknown)}")
        return self


class AppConfig(_Strict):
    seed: int
    paths: PathsConfig
    generation: GenerationDefaults
    models: ModelsConfig
    budget: BudgetConfig
    datasets: dict[str, DatasetSettings]
    retrieval: RetrievalConfig
    graph: GraphConfig
    router: RouterConfig


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; nested dicts merge, other values replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return data


def load_config(
    overrides: Sequence[Path] = (),
    *,
    config_dir: Path = CONFIG_DIR,
    root: Path = PROJECT_ROOT,
) -> AppConfig:
    """Load base, models and budget configs, apply ``overrides`` in order, and validate."""
    raw = load_yaml(config_dir / "base.yaml")
    raw["models"] = load_yaml(config_dir / "models.yaml")
    raw["budget"] = load_yaml(config_dir / "budget.yaml")
    raw["datasets"] = load_yaml(config_dir / "datasets.yaml")
    raw["retrieval"] = load_yaml(config_dir / "retrieval.yaml")
    raw["graph"] = load_yaml(config_dir / "graph.yaml")
    raw["router"] = load_yaml(config_dir / "router.yaml")
    for override in overrides:
        raw = deep_merge(raw, load_yaml(override))

    config = AppConfig.model_validate(raw)
    return config.model_copy(update={"paths": config.paths.resolved(root)})


def _absolute(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path
