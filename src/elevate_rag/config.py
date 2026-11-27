"""Validated, hashable configuration.

A pipeline config fully determines an index and a query pipeline. YAML files may `extends:` a
base file and experiments apply dotted-path overrides (`retrieval.final_k: 5`). The config hash
(over the *resolved* config) is recorded in every result row (FR-O2).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from elevate_rag.utils import ConfigError, stable_hash


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_Strict):
    strategy: Literal["sentence", "paragraph"] = "sentence"
    chunk_tokens: int = Field(512, ge=32, le=8192)
    overlap_tokens: int = Field(64, ge=0)
    encoding: str = "o200k_base"
    prepend_title: bool = True  # chunk text = "title\ntext" so every retriever sees the title

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


class ProviderConfig(_Strict):
    """One model role. `provider` selects the implementation; `model` is the exact model ID."""

    provider: str
    model: str
    revision: str | None = None
    family: str | None = None  # e.g. "claude", "qwen", "gemma" — used for judge independence
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int = Field(4, ge=1, le=256)
    timeout_s: float = Field(120.0, gt=0)
    max_retries: int = Field(4, ge=0, le=10)
    batch_size: int = Field(32, ge=1)
    # USD per million tokens; None = unknown (cost reported as missing, not zero).
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    dimension: int | None = None  # embedders: expected vector size (validated, FR-M5)

    @property
    def identity(self) -> dict[str, Any]:
        """Fields that change model outputs (cache keys, index versions)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "params": self.params,
        }


class ModelsConfig(_Strict):
    embedder: ProviderConfig
    reranker: ProviderConfig | None = None
    extractor: ProviderConfig | None = None
    keyword: ProviderConfig | None = None  # defaults to extractor
    generator: ProviderConfig
    router_llm: ProviderConfig | None = None
    judge: ProviderConfig | None = None


class VectorStoreConfig(_Strict):
    backend: Literal["exact", "faiss", "qdrant", "milvus", "pgvector"] = "exact"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    url_env: str | None = None  # env var holding the connection URL/DSN
    collection_prefix: str = "elevate"


class GraphStoreConfig(_Strict):
    backend: Literal["networkx", "neo4j", "memgraph", "postgres"] = "networkx"
    url_env: str | None = None
    user_env: str | None = None
    password_env: str | None = None


class LightRAGConfig(_Strict):
    """Maps 1:1 onto LightRAG constructor / QueryParam fields (D-05)."""

    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    entity_extract_max_gleaning: int = Field(1, ge=0, le=3)
    # Output cap for one extraction call. Small local models can loop until the cap; keep
    # prompt (~2-3.5k tokens) + this below the server's context window or it truncates.
    extract_max_tokens: int = Field(4096, ge=256)
    force_llm_summary_on_merge: int = Field(8, ge=1)
    top_k: int = Field(40, ge=1)
    chunk_top_k: int = Field(20, ge=1)
    cosine_better_than_threshold: float = 0.2
    max_parallel_insert: int = Field(2, ge=1)
    llm_max_async: int = Field(4, ge=1)
    vector_storage_kwargs: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(_Strict):
    candidates: int = Field(50, ge=1)  # top-N before reranking (plan: 50)
    final_k: int = Field(8, ge=1, le=50)  # after reranking (plan: 5–10)
    rrf_k: int = Field(60, ge=1)
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    context_budget: int = Field(8192, ge=256)  # reference budget; A3 ablates 4k/8k/16k
    kg_share: float = Field(0.5, ge=0.0, le=1.0)
    graph_depth: Literal[1, 2] = 1
    max_kg_items: int = Field(60, ge=0)


class ResolutionConfig(_Strict):
    enabled: bool = True
    similarity_threshold: float = Field(0.92, ge=0.5, le=1.0)
    max_merges: int = Field(10_000, ge=0)


class CommunityConfig(_Strict):
    enabled: bool = False
    resolution: float = 1.0
    min_size: int = 3
    max_reports: int = 200
    top_reports: int = 5


class RouterConfig(_Strict):
    level: Literal["static", "rules", "classifier", "llm", "fallback"] = "static"
    static_mode: str = "vector"
    modes: list[str] = Field(default_factory=lambda: ["vector", "local", "global", "mix"])
    # query type -> mode used by R1/R2/R3 (query-type labelling scheme)
    type_to_mode: dict[str, str] = Field(
        default_factory=lambda: {
            "single_hop": "vector",
            "null": "vector",
            "other": "vector",
            "multi_hop": "local",
            "comparison": "local",
            "temporal": "mix",
            "global": "global",
            "creative": "global",
        }
    )
    classifier_path: str | None = None
    fallback_cheap_mode: str = "vector"
    fallback_escalate_mode: str = "mix"
    fallback_threshold: float = 0.0  # tuned on dev; compared with top rerank score
    fallback_on_refusal: bool = True


class BudgetConfig(_Strict):
    max_usd: float = Field(0.0, ge=0.0)  # hard cap; 0 = no paid calls allowed
    pilot_fraction: float = Field(0.05, gt=0.0, le=1.0)


LOCATION_FIELDS = frozenset({"data_dir", "cache_dir", "index_dir"})
# Per-model settings that say *where/how* a model is reached, not what it outputs. A cluster job
# gets a different node (and vLLM URL) on every submission; these must not change run ids.
OPERATIONAL_MODEL_FIELDS = frozenset(
    {"base_url", "api_key_env", "max_concurrency", "timeout_s", "max_retries"}
)
OPERATIONAL_LIGHTRAG_FIELDS = frozenset({"llm_max_async", "max_parallel_insert"})


def default_dir(env_var: str, fallback: str) -> str:
    """A storage directory from the environment (unset or empty → ``fallback``)."""
    return os.getenv(env_var) or fallback


class PipelineConfig(_Strict):
    name: str
    dataset: str | None = None
    chunking: ChunkingConfig = Field(default_factory=lambda: ChunkingConfig())
    models: ModelsConfig
    vector_store: VectorStoreConfig = Field(default_factory=lambda: VectorStoreConfig())
    graph_store: GraphStoreConfig = Field(default_factory=lambda: GraphStoreConfig())
    lightrag: LightRAGConfig = Field(default_factory=lambda: LightRAGConfig())
    retrieval: RetrievalConfig = Field(default_factory=lambda: RetrievalConfig())
    resolution: ResolutionConfig = Field(default_factory=lambda: ResolutionConfig())
    community: CommunityConfig = Field(default_factory=lambda: CommunityConfig())
    router: RouterConfig = Field(default_factory=lambda: RouterConfig())
    budget: BudgetConfig = Field(default_factory=lambda: BudgetConfig())
    # Storage locations. Defaults come from the environment so a deployment (e.g. an EC2 data
    # volume) can relocate them without editing configs. They are NOT part of config_hash().
    data_dir: str = Field(default_factory=lambda: default_dir("ELEVATE_DATA_DIR", "data"))
    cache_dir: str = Field(default_factory=lambda: default_dir("ELEVATE_CACHE_DIR", "data/cache"))
    index_dir: str = Field(default_factory=lambda: default_dir("ELEVATE_INDEX_DIR", "indexes"))
    demo: bool = False  # offline demonstration config — results are not research results

    def config_hash(self) -> str:
        """Identity of the *experimental* configuration. Storage locations and operational model
        settings (endpoint URL, concurrency, timeouts) are excluded, so a run keeps its id (and
        stays resumable) when data moves or a cluster job lands on another node."""
        data = self.model_dump(mode="json", exclude=set(LOCATION_FIELDS))
        for role in data["models"].values():
            if isinstance(role, dict):
                for k in OPERATIONAL_MODEL_FIELDS:
                    role.pop(k, None)
        for k in OPERATIONAL_LIGHTRAG_FIELDS:
            data["lightrag"].pop(k, None)
        return stable_hash(data)

    def index_key(self) -> str:
        """Identity of the *index*: corpus, chunking, embedder, extractor, resolution, community.
        Changing any of these yields a new index directory (FR-M5)."""
        m = self.models
        return stable_hash(
            {
                "dataset": self.dataset,
                "chunking": self.chunking.model_dump(),
                "embedder": m.embedder.identity,
                "extractor": m.extractor.identity if m.extractor else None,
                "lightrag": self.lightrag.model_dump(
                    include={
                        "entity_extract_max_gleaning",
                        "extract_max_tokens",
                        "force_llm_summary_on_merge",
                        "kv_storage",
                        "vector_storage",
                        "graph_storage",
                    }
                ),
                "resolution": self.resolution.model_dump(),
                "community": self.community.model_dump(),
            }
        )


# --------------------------------------------------------------------------- loading


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-path overrides, e.g. {"retrieval.final_k": 5}."""
    out = copy.deepcopy(data)
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = node[p] = {}
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {dotted!r}: {p!r} is not a mapping")
            node = nxt
        node[parts[-1]] = copy.deepcopy(value)
    return out


def load_yaml_with_extends(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular `extends` involving {path}")
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parents = data.pop("extends", None)
    if parents:
        merged: dict[str, Any] = {}
        for parent in [parents] if isinstance(parents, str) else parents:
            merged = deep_merge(
                merged, load_yaml_with_extends(path.parent / parent, _seen | {path})
            )
        data = merge_child_config(merged, data)
    return data


def merge_child_config(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """`extends` semantics: deep merge, except that a model role the child defines with its own
    `model:` *replaces* the parent's role. Otherwise provider-specific settings leak across
    providers (e.g. Anthropic's `effort` reaching an OpenAI-compatible server)."""
    parent = copy.deepcopy(parent)
    for role, spec in (child.get("models") or {}).items():
        if isinstance(spec, dict) and "model" in spec:
            (parent.get("models") or {}).pop(role, None)
    return deep_merge(parent, child)


def _expand_env(obj: Any) -> Any:
    """Expand ${VAR}; an unset variable is left as-is and rejected where it is *used*
    (see `require_resolved`), so configs that merely mention it still load."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def require_resolved(value: str | None, what: str) -> str | None:
    if value is not None and "${" in value:
        raise ConfigError(f"{what}: environment variable in {value!r} is not set")
    return value


def load_pipeline_config(
    path: Path | str, overrides: dict[str, Any] | None = None
) -> PipelineConfig:
    data = load_yaml_with_extends(Path(path))
    if overrides:
        data = apply_overrides(data, overrides)
    try:
        return PipelineConfig.model_validate(_expand_env(data))
    except ValueError as exc:
        raise ConfigError(f"invalid pipeline config {path}: {exc}") from exc
