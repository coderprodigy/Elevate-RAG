"""E4 Level A vector benchmark at matched recall (plan "Database comparison plan").

For each store: build (timed) → sweep ``ef`` (recall@k vs exact FAISS, single-client QPS) →
pick the smallest ``ef`` reaching ``target_recall`` → at that point measure latency
p50/p95/p99, QPS at several client counts, filtered search, write-to-searchable delay and memory.
Timings use a warm-up pass and the median of ``repeats``.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import faiss
import numpy as np

from hgrag.bench.vector_stores import VectorStore
from hgrag.eval.metrics.efficiency import percentile
from hgrag.index.embedders import Vectors


@dataclass(frozen=True)
class VectorBenchSettings:
    k: int = 10
    target_recall: float = 0.95
    ef_sweep: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    concurrency: tuple[int, ...] = (1, 8, 32)
    warmup_queries: int = 1000
    repeats: int = 3
    filter_group: int = 3
    n_groups: int = 10


@dataclass
class StoreReport:
    store: str
    n_vectors: int
    build_seconds: float
    curve: list[dict[str, float]]  # per ef: recall, single-client QPS
    matched_ef: int
    matched_recall: float
    reached_target: bool
    latency_ms: dict[str, float]  # p50 / p95 / p99 at the matched ef
    qps: dict[int, float]  # clients → queries per second
    filtered_recall: float
    filtered_p50_ms: float
    write_to_searchable_ms: float
    memory_mb: float | None
    notes: list[str] = field(default_factory=list)


def exact_top_k(
    vectors: Vectors,
    queries: Vectors,
    k: int,
    groups: np.ndarray | None = None,
    group: int | None = None,
) -> list[list[int]]:
    """Ground truth: exact inner-product search (optionally within one group)."""
    rows = (
        np.arange(len(vectors))
        if group is None or groups is None
        else np.flatnonzero(groups == group)
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors[rows], dtype=np.float32))
    _, found = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return [[int(rows[i]) for i in row if i != -1] for row in found]


def recall_at_k(results: Sequence[Sequence[int]], truth: Sequence[Sequence[int]], k: int) -> float:
    hits = sum(len(set(r[:k]) & set(t[:k])) for r, t in zip(results, truth, strict=True))
    return hits / max(sum(min(k, len(t)) for t in truth), 1)


def _timed_searches(
    store: VectorStore, queries: Vectors, k: int, ef: int, group: int | None = None
) -> tuple[list[list[int]], list[float]]:
    results, latencies = [], []
    for query in queries:
        started = time.perf_counter()
        results.append(store.search(query, k, ef, group))
        latencies.append((time.perf_counter() - started) * 1000)
    return results, latencies


def _qps(store: VectorStore, queries: Vectors, k: int, ef: int, clients: int) -> float:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        list(pool.map(lambda q: store.search(q, k, ef), queries))
    return len(queries) / (time.perf_counter() - started)


def benchmark_store(
    store: VectorStore,
    vectors: Vectors,
    queries: Vectors,
    settings: VectorBenchSettings,
    *,
    memory_probe: Callable[[], float | None] = lambda: None,
) -> StoreReport:
    groups = np.arange(len(vectors)) % settings.n_groups
    truth = exact_top_k(vectors, queries, settings.k)

    started = time.perf_counter()
    store.build(vectors, groups.tolist())
    build_seconds = time.perf_counter() - started

    warm = queries[: min(len(queries), settings.warmup_queries)]
    _timed_searches(store, warm, settings.k, max(settings.ef_sweep))

    curve = []
    matched: tuple[int, float] | None = None
    for ef in sorted(settings.ef_sweep):
        results, latencies = _timed_searches(store, queries, settings.k, ef)
        recall = recall_at_k(results, truth, settings.k)
        curve.append(
            {
                "ef": ef,
                "recall": round(recall, 4),
                "qps": round(1000 / statistics.fmean(latencies), 1),
            }
        )
        if matched is None and recall >= settings.target_recall:
            matched = (ef, recall)
    reached = matched is not None
    ef, matched_recall = matched or (max(settings.ef_sweep), curve[-1]["recall"])

    latency_runs = [
        _timed_searches(store, queries, settings.k, ef)[1] for _ in range(settings.repeats)
    ]
    latency = {
        f"p{q}": round(statistics.median(percentile(run, q) for run in latency_runs), 3)
        for q in (50, 95, 99)
    }
    qps = {
        clients: round(
            statistics.median(
                _qps(store, queries, settings.k, ef, clients) for _ in range(settings.repeats)
            ),
            1,
        )
        for clients in settings.concurrency
    }

    filtered_truth = exact_top_k(vectors, queries, settings.k, groups, settings.filter_group)
    filtered, filtered_latency = _timed_searches(
        store, queries, settings.k, ef, settings.filter_group
    )

    new_id = len(vectors)
    probe = queries[0]
    store.insert(new_id, probe, settings.filter_group)
    started = time.perf_counter()
    while new_id not in store.search(probe, settings.k, ef) and time.perf_counter() - started < 30:
        time.sleep(0.01)
    write_ms = (time.perf_counter() - started) * 1000

    return StoreReport(
        store=store.name,
        n_vectors=len(vectors),
        build_seconds=round(build_seconds, 2),
        curve=curve,
        matched_ef=ef,
        matched_recall=round(matched_recall, 4),
        reached_target=reached,
        latency_ms=latency,
        qps=qps,
        filtered_recall=round(recall_at_k(filtered, filtered_truth, settings.k), 4),
        filtered_p50_ms=round(percentile(filtered_latency, 50), 3),
        write_to_searchable_ms=round(write_ms, 1),
        memory_mb=memory_probe(),
        notes=[] if reached else [f"target recall {settings.target_recall} not reached in sweep"],
    )
