"""Build an index for one dataset: chunks → gold-chunk map → BM25 → embeddings → HNSW.

Idempotent: the manifest records a hash of the chunk IDs; existing parts that match are reused,
so re-running only adds what is missing (e.g. a second embedder).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hgrag.config import AppConfig
from hgrag.core.hashing import stable_hash
from hgrag.data.io import dataset_files, load_corpus, load_queries, read_jsonl, write_jsonl
from hgrag.index.bm25_index import BM25Index
from hgrag.index.chunking import Chunk, chunk_documents
from hgrag.index.embedders import Embedder
from hgrag.index.gold import map_gold_chunks
from hgrag.index.models import load_embedder
from hgrag.index.paths import IndexPaths
from hgrag.index.vector_index import FaissIndex

logger = logging.getLogger(__name__)

EmbedderLoader = Callable[[str], Embedder]


class IndexMissingError(FileNotFoundError):
    """An index part is missing; run ``hgrag index build``."""


@dataclass
class IndexManifest:
    dataset: str
    chunker: str
    chunk_set_hash: str
    n_chunks: int
    mean_chunk_tokens: float
    gold_questions: int
    evidence_mapped_rate: float
    embedders: dict[str, dict[str, object]] = field(default_factory=dict)


def default_chunker(config: AppConfig, dataset: str) -> str:
    return config.retrieval.dataset_chunker.get(dataset, "sent512")


def build_index(
    config: AppConfig,
    dataset: str,
    *,
    chunker: str | None = None,
    embedders: list[str] | None = None,
    embedder_loader: EmbedderLoader | None = None,
) -> IndexManifest:
    chunker = chunker or default_chunker(config, dataset)
    paths = IndexPaths.of(config, dataset, chunker)
    manifest = _build_chunks(config, dataset, chunker, paths)
    loader = embedder_loader or (lambda name: load_embedder(name, config.retrieval.embedders[name]))
    for name in embedders or []:
        if name in manifest.embedders and paths.hnsw(name).is_file():
            logger.info("%s/%s: embeddings for %s already built", dataset, chunker, name)
            continue
        manifest.embedders[name] = _build_embeddings(config, paths, name, loader(name))
        _write_manifest(paths, manifest)
    return manifest


def load_chunks(paths: IndexPaths) -> list[Chunk]:
    if not paths.chunks.is_file():
        raise IndexMissingError(f"no index at {paths.root}; run: hgrag index build")
    return read_jsonl(paths.chunks, Chunk)


def load_manifest(paths: IndexPaths) -> IndexManifest:
    if not paths.manifest.is_file():
        raise IndexMissingError(f"no index at {paths.root}; run: hgrag index build")
    return IndexManifest(**json.loads(paths.manifest.read_text(encoding="utf-8")))


def _build_chunks(
    config: AppConfig, dataset: str, chunker: str, paths: IndexPaths
) -> IndexManifest:
    files = dataset_files(config, dataset)
    chunks = list(chunk_documents(load_corpus(files), config.retrieval.chunkers[chunker]))
    chunk_set_hash = stable_hash([c.chunk_id for c in chunks])

    if paths.manifest.is_file():
        existing = load_manifest(paths)
        if existing.chunk_set_hash == chunk_set_hash:
            logger.info("%s/%s: chunks up to date (%d)", dataset, chunker, existing.n_chunks)
            return existing
        logger.warning("%s/%s: chunks changed; rebuilding the whole index", dataset, chunker)

    paths.root.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.chunks, chunks)
    gold = map_gold_chunks(load_queries(files), chunks)
    paths.gold_chunks.write_text(json.dumps(gold.gold_chunks), encoding="utf-8")
    ids = [c.chunk_id for c in chunks]
    BM25Index.build([c.indexed_text for c in chunks], ids).save(paths.bm25)

    manifest = IndexManifest(
        dataset=dataset,
        chunker=chunker,
        chunk_set_hash=chunk_set_hash,
        n_chunks=len(chunks),
        mean_chunk_tokens=round(sum(c.n_tokens for c in chunks) / max(len(chunks), 1), 1),
        gold_questions=len(gold.gold_chunks),
        evidence_mapped_rate=round(gold.evidence_mapped_rate, 4),
    )
    _write_manifest(paths, manifest)
    logger.info(
        "%s/%s: %d chunks, gold for %d questions, evidence mapped %.1f%%",
        dataset,
        chunker,
        manifest.n_chunks,
        manifest.gold_questions,
        100 * manifest.evidence_mapped_rate,
    )
    return manifest


def _build_embeddings(
    config: AppConfig, paths: IndexPaths, name: str, embedder: Embedder
) -> dict[str, object]:
    chunks = load_chunks(paths)
    logger.info("embedding %d chunks with %s", len(chunks), name)
    vectors = embedder.embed_documents([c.indexed_text for c in chunks])
    paths.embedding_dir(name).mkdir(parents=True, exist_ok=True)
    np.save(paths.embeddings(name), vectors)
    FaissIndex.build(vectors, [c.chunk_id for c in chunks], config.retrieval.hnsw).save(
        paths.hnsw(name)
    )
    spec = config.retrieval.embedders.get(name)
    return {"model": spec.model if spec else name, "dim": int(vectors.shape[1])}


def _write_manifest(paths: IndexPaths, manifest: IndexManifest) -> None:
    paths.manifest.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")
