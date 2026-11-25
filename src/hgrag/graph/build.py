"""Build a LightRAG graph over our chunks: full build, or a seeded sample to measure cost first.

Cost-first workflow (plan Phase 4, step 2):
1. ``estimate_build`` gives an upper bound from chunk counts and prices (no calls).
2. ``build_graph(sample=True)`` indexes ``sample_fraction`` of the chunks and records the real
   spend; the manifest extrapolates it to the full corpus.
3. After approval, ``build_graph(sample=False)``. Re-running resumes (processed documents skip).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass

from hgrag.config import AppConfig
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, load_corpus
from hgrag.graph.lightrag_adapter import (
    EXTRACT_ROLE,
    ChunkRegistry,
    GraphLocation,
    build_lightrag,
)
from hgrag.index.build import load_chunks, load_manifest
from hgrag.index.chunking import Chunk
from hgrag.index.embedders import Embedder
from hgrag.index.paths import IndexPaths
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.ledger import UsageLedger
from hgrag.llm.pricing import cost_usd

logger = logging.getLogger(__name__)
MANIFEST = "hgrag_manifest.json"
GatewayFactory = Callable[[AppConfig, RunContext], LLMGateway]


@dataclass
class GraphManifest:
    dataset: str
    chunker: str
    extractor: str
    embedder: str
    backend: str
    sample: bool
    chunk_set_hash: str
    n_documents: int
    n_chunks: int
    corpus_chunks: int
    status_counts: dict[str, int]
    spent_usd: float  # cumulative across resumes (ledger, this build's run ID)
    seconds: float
    extrapolated_full_usd: float | None  # sample builds only


@dataclass(frozen=True)
class BuildEstimate:
    n_chunks: int
    upper_bound_usd: float
    sample_chunks: int
    sample_upper_bound_usd: float


def estimate_build(config: AppConfig, dataset: str, chunker: str, extractor: str) -> BuildEstimate:
    """Upper bound from configured per-chunk token bounds and the extractor's prices."""
    n = load_manifest(IndexPaths.of(config, dataset, chunker)).n_chunks
    extraction = config.graph.extraction
    price = config.models.models[extractor].price
    per_chunk = cost_usd(
        price, extraction.est_input_tokens_per_chunk, extraction.est_output_tokens_per_chunk, 0
    )
    sample = _sample_size(n, extraction.sample_fraction)
    return BuildEstimate(n, n * per_chunk, sample, sample * per_chunk)


def select_chunks(chunks: list[Chunk], fraction: float | None, seed: int) -> list[Chunk]:
    if fraction is None:
        return chunks
    picked = sorted(
        random.Random(seed).sample(range(len(chunks)), _sample_size(len(chunks), fraction))
    )
    return [chunks[i] for i in picked]


def build_graph(
    config: AppConfig,
    location: GraphLocation,
    *,
    embedder: Embedder,
    gateway_factory: GatewayFactory = build_gateway,
) -> GraphManifest:
    return asyncio.run(_build(config, location, embedder, gateway_factory))


async def _build(
    config: AppConfig,
    location: GraphLocation,
    embedder: Embedder,
    gateway_factory: GatewayFactory,
) -> GraphManifest:
    paths = IndexPaths.of(config, location.dataset, location.chunker)
    all_chunks = load_chunks(paths)
    fraction = config.graph.extraction.sample_fraction if location.sample else None
    chunks = select_chunks(all_chunks, fraction, config.seed)

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    full_text = {d.doc_id: d.text for d in load_corpus(dataset_files(config, location.dataset))}
    complete: dict[str, int] = defaultdict(int)
    for chunk in all_chunks:
        complete[chunk.doc_id] += 1

    registry = ChunkRegistry()
    doc_ids, texts = [], []
    for doc_id, doc_chunks in sorted(by_doc.items()):
        # A document with only some chunks sampled is inserted as just those chunks' text.
        text = (
            full_text[doc_id]
            if len(doc_chunks) == complete[doc_id]
            else "\n\n".join(c.text for c in doc_chunks)
        )
        registry.register(text, doc_chunks)
        doc_ids.append(doc_id)
        texts.append(text)

    context = RunContext(run_id=location.run_id, experiment="graph", phase="index")
    gateway = gateway_factory(config, context)
    rag = build_lightrag(
        config,
        location,
        gateway=gateway,
        embedder=embedder,
        registry=registry,
        extractor_model_key=location.extractor,
    )
    started = time.perf_counter()
    await rag.initialize_storages()
    try:
        await rag.ainsert(texts, ids=doc_ids, file_paths=doc_ids)
        counts = await rag.doc_status.get_status_counts()
    finally:
        await rag.finalize_storages()

    spent = UsageLedger(config.paths.ledger_db).total_cost(location.run_id)
    manifest = GraphManifest(
        dataset=location.dataset,
        chunker=location.chunker,
        extractor=location.extractor,
        embedder=embedder.name,
        backend=location.backend,
        sample=location.sample,
        chunk_set_hash=load_manifest(paths).chunk_set_hash,
        n_documents=len(doc_ids),
        n_chunks=len(chunks),
        corpus_chunks=len(all_chunks),
        status_counts={str(k): int(v) for k, v in counts.items()},
        spent_usd=round(spent, 6),
        seconds=round(time.perf_counter() - started, 1),
        extrapolated_full_usd=(
            round(spent * len(all_chunks) / max(len(chunks), 1), 4) if location.sample else None
        ),
    )
    (location.working_dir(config) / MANIFEST).write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def load_graph_manifest(config: AppConfig, location: GraphLocation) -> GraphManifest:
    path = location.working_dir(config) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"no graph at {path.parent}; run: hgrag graph build")
    return GraphManifest(**json.loads(path.read_text(encoding="utf-8")))


def default_extractor(config: AppConfig) -> str:
    return config.models.model_key_for_role(EXTRACT_ROLE)


def _sample_size(n: int, fraction: float) -> int:
    return max(1, min(n, round(n * fraction)))
