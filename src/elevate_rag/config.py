"""Validated immutable configuration for a pipeline or one controlled variant."""

import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .data import stable_hash

VECTOR_MODES = ("V0", "V1", "V2", "V3", "V4")
GRAPH_MODES = ("naive", "local", "global", "hybrid", "mix")
MODES = (*VECTOR_MODES, *GRAPH_MODES)
ROLES = ("embedding", "reranking", "extraction", "keywords", "generation", "router", "judge")
PROVIDERS = {"fixture", "sentence-transformers", "openai-compatible", "jina-rerank", "disabled"}
LIGHT_VECTOR = {
    "nano": "NanoVectorDBStorage",
    "qdrant": "QdrantVectorDBStorage",
    "milvus": "MilvusVectorDBStorage",
    "pgvector": "PGVectorStorage",
    "faiss": "FaissVectorDBStorage",
}
LIGHT_GRAPH = {
    "networkx": "NetworkXStorage",
    "neo4j": "Neo4JStorage",
    "memgraph": "MemgraphStorage",
    "postgres": "PGGraphStorage",
}


@dataclass(frozen=True, slots=True)
class Model:
    provider: str
    model: str
    revision: str
    base_url: str = ""
    key_env: str = ""
    family: str = ""
    dimensions: int = 0
    context_window: int = 4096
    max_output: int = 256
    seed: int | None = None
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    paid: bool = False
    device: str = "cpu"
    query_prompt: str = ""

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS or not self.model.strip() or not self.revision.strip():
            raise ValueError("model needs a supported provider, ID and revision")
        if self.provider == "sentence-transformers" and not re.fullmatch(
            r"[a-f0-9]{40}", self.revision
        ):
            raise ValueError("Hugging Face models require an exact 40-character revision")
        if self.provider in {"openai-compatible", "jina-rerank"}:
            url = urlsplit(self.base_url)
            loopback = url.hostname in {"localhost", "127.0.0.1", "::1"}
            if (
                not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or (url.scheme != "https" and (not loopback or url.scheme != "http"))
            ):
                raise ValueError(
                    "provider requires HTTPS or a loopback HTTP endpoint without credentials"
                )
            if not loopback and (not self.paid or not self.key_env):
                raise ValueError(
                    "remote API models require a credential environment variable and spend tracking"
                )
            if not loopback and self.input_per_million == self.output_per_million == 0:
                raise ValueError("remote API models require a configured price")
        prices = (self.input_per_million, self.output_per_million)
        if any(not math.isfinite(price) or price < 0 for price in prices):
            raise ValueError("model prices must be finite and nonnegative")
        if self.dimensions < 0 or not 0 < self.max_output < self.context_window:
            raise ValueError("invalid model dimensions or token limits")

    @property
    def identity(self) -> str:
        return f"{self.model}@{self.revision}"


@dataclass(frozen=True, slots=True)
class Settings:
    synthetic: bool = False
    seed: int = 42
    chunk_size: int = 512
    overlap: int = 64
    paragraph_chunks: bool = False
    candidates: int = 50
    rerank_k: int = 8
    rrf_k: int = 60
    context_budget: int = 4096
    chunk_tokenizer_model: str = ""
    chunk_tokenizer_revision: str = ""
    context_tokenizer_model: str = ""
    context_tokenizer_revision: str = ""
    local_models_only: bool = True
    embedding_batch: int = 64
    concurrency: int = 4
    timeout: float = 60.0
    retries: int = 2
    spending_cap: float = 0.0
    cache_retrieval: bool = True
    cache_generation: bool = False
    cache_judge: bool = True
    ragas_metrics: bool = False
    langfuse_tracing: bool = False
    vector_store: str = "exact"
    graph_vector_store: str = "nano"
    graph_store: str = "networkx"
    namespace: str = "elevate_atlas"
    hnsw_m: int = 16
    ef_construction: int = 200
    ef_search: int = 100
    graph_depth: int = 1
    graph_extractor: str = "lightrag"
    graph_merge_threshold: float | None = None
    community: bool = False
    fallback_threshold: float = 0.0
    initial_route: str = "V4"
    models: dict[str, Model] = field(default_factory=dict)
    rules: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("invalid chunk size or overlap")
        if not 0 < self.rerank_k <= self.candidates or self.rrf_k <= 0 or self.context_budget <= 0:
            raise ValueError("invalid retrieval or context parameters")
        if (
            min(self.embedding_batch, self.concurrency, self.timeout) <= 0
            or not 0 <= self.retries <= 5
        ):
            raise ValueError("invalid batch, timeout, concurrency or retry limit")
        if not math.isfinite(self.spending_cap) or self.spending_cap < 0:
            raise ValueError("spending cap must be finite and nonnegative")
        if self.vector_store not in {"exact", "faiss", "qdrant", "milvus", "pgvector"}:
            raise ValueError("unknown vector storage backend")
        if self.graph_store not in LIGHT_GRAPH or self.graph_vector_store not in LIGHT_VECTOR:
            raise ValueError("unknown LightRAG storage backend")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", self.namespace):
            raise ValueError("invalid storage namespace")
        if min(self.hnsw_m, self.ef_search, self.ef_construction) <= 0 or self.graph_depth not in {
            1,
            2,
        }:
            raise ValueError("invalid index settings or graph depth")
        if self.graph_extractor not in {"lightrag", "llamaindex-schema"}:
            raise ValueError("unknown graph extraction method")
        if self.graph_merge_threshold is not None and not 0 < self.graph_merge_threshold <= 1:
            raise ValueError("invalid entity merge threshold")
        if self.initial_route not in MODES or self.initial_route in {"mix", "V0"}:
            raise ValueError("fallback initial route must retrieve without using mix")
        if not math.isfinite(self.fallback_threshold):
            raise ValueError("fallback confidence threshold must be finite")
        if set(self.models) - set(ROLES):
            raise ValueError("unknown model role")
        if set(self.rules) - set(MODES):
            raise ValueError("unsupported rule route")
        for patterns in self.rules.values():
            for pattern in patterns:
                re.compile(pattern)
        if self.models.get("embedding") and self.models["embedding"].dimensions <= 0:
            raise ValueError("embedding model must declare its expected dimensions")
        for role, model in self.models.items():
            if model.provider == "fixture" and not self.synthetic:
                raise ValueError(f"{role}: fixture providers require a synthetic configuration")
            if model.provider == "jina-rerank" and role != "reranking":
                raise ValueError("Jina rerank provider must be assigned to reranking")
            if model.provider == "sentence-transformers" and role not in {"embedding", "reranking"}:
                raise ValueError(f"{role}: sentence-transformers supports embedding and reranking")
        if not self.synthetic:
            for tokenizer_model, revision in (
                (self.chunk_tokenizer_model, self.chunk_tokenizer_revision),
                (self.context_tokenizer_model, self.context_tokenizer_revision),
            ):
                if not tokenizer_model or not re.fullmatch(r"[a-f0-9]{40}", revision):
                    raise ValueError(
                        "real runs require frozen chunk and context tokenizer revisions"
                    )
        judge = self.models.get("judge")
        generator = self.models.get("generation")
        if judge and generator and judge.provider not in {"fixture", "disabled"}:
            if (
                not judge.family
                or not generator.family
                or judge.family.casefold() == generator.family.casefold()
            ):
                raise ValueError("judge must use an identified family distinct from the generator")

    def model(self, role: str) -> Model:
        if role not in self.models or self.models[role].provider == "disabled":
            raise ValueError(f"configure an enabled model for role {role}")
        return self.models[role]

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def read(cls, path: Path) -> "Settings":
        with path.open("rb") as stream:
            values: dict[str, Any] = (
                json.load(stream) if path.suffix == ".json" else tomllib.load(stream)
            )
        values["models"] = {role: Model(**row) for role, row in values.get("models", {}).items()}
        return cls(**values)
