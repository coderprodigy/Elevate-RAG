"""Level-A database micro-benchmarks (FR-DB4, FR-DB5; plan §Database comparison).

Vector protocol: ground truth = exact inner-product search. For each engine, sweep the search
breadth (`ef`) and record recall@10 vs QPS; the *matched point* is the smallest ef reaching
recall@10 ≥ 0.95. At that point measure QPS at 1/8/32 concurrent clients, p50/p95/p99 latency,
filtered-search latency and write-to-searchable delay. 1,000 warm-up queries precede timing and
each timed measurement is repeated 3× (median reported).

Memory: in-process engines report the process RSS delta; server engines report the container's
memory from `docker stats` when a container name is given (otherwise "not measured").
"""

from __future__ import annotations

import random
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import psutil

from elevate_rag.storage.graph import Edge, GraphStore, Node
from elevate_rag.storage.vector import VectorStore, wait_until_searchable


def _percentiles(lat: Sequence[float]) -> dict[str, float]:
    a = np.asarray(lat) * 1000
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
    }


def exact_ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    scores = queries @ base.T
    return [list(np.argsort(-row, kind="stable")[:k]) for row in scores]


def recall_at(results: list[list[str]], truth: list[list[str]], k: int) -> float:
    return float(
        np.mean([len(set(r[:k]) & set(t[:k])) / k for r, t in zip(results, truth, strict=True)])
    )


def _timed_queries(
    store: VectorStore, queries: np.ndarray, k: int, flt: dict[str, Any] | None = None
) -> tuple[list[list[str]], list[float]]:
    res, lat = [], []
    for q in queries:
        t0 = time.perf_counter()
        hits = store.search(q, k, flt)
        lat.append(time.perf_counter() - t0)
        res.append([h[0] for h in hits])
    return res, lat


def concurrent_qps(
    store: VectorStore, queries: np.ndarray, k: int, clients: int, repeats: int = 3
) -> float:
    def worker(qs: np.ndarray) -> None:
        for q in qs:
            store.search(q, k)

    runs = []
    for _ in range(repeats):
        parts = np.array_split(queries, clients)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(clients) as ex:
            list(ex.map(worker, parts))
        runs.append(len(queries) / (time.perf_counter() - t0))
    return float(statistics.median(runs))


def docker_memory_mb(container: str | None) -> float | None:
    if not container:
        return None
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    used = out.split("/")[0].strip()
    units = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "kB": 1 / 1000, "MB": 1.0, "GB": 1000.0}
    for u, f in units.items():
        if used.endswith(u):
            return float(used[: -len(u)]) * f
    return None


def bench_vector_store(
    make_store: Callable[[], VectorStore],
    ids: list[str],
    base: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    target_recall: float = 0.95,
    ef_values: Sequence[int] = (16, 32, 64, 128, 256, 512),
    clients: Sequence[int] = (1, 8, 32),
    warmup: int = 1000,
    container: str | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    store = make_store()
    meta = [{"source": f"s{i % 4}"} for i in range(len(ids))]
    t0 = time.perf_counter()
    for s in range(0, len(ids), 1000):
        store.add(ids[s : s + 1000], base[s : s + 1000], meta[s : s + 1000])
    build_s = time.perf_counter() - t0
    truth_idx = exact_ground_truth(base, queries, k)
    truth = [[ids[i] for i in row] for row in truth_idx]

    rng = np.random.default_rng(seed)
    warm = queries[rng.integers(0, len(queries), size=warmup)] if len(queries) else queries
    for q in warm:
        store.search(q, k)

    curve = []
    for ef in ef_values:
        store.set_search_param(ef)
        reps = [_timed_queries(store, queries, k) for _ in range(3)]
        rec = recall_at(reps[0][0], truth, k)
        qps = statistics.median(len(queries) / sum(lat) for _, lat in reps)
        curve.append({"ef": ef, "recall@10": rec, "qps_1_client": qps, **_percentiles(reps[0][1])})
    matched = next((c for c in curve if c["recall@10"] >= target_recall), None)
    out: dict[str, Any] = {
        "backend": store.backend,
        "n_vectors": len(ids),
        "dim": base.shape[1],
        "build_s": build_s,
        "curve": curve,
        "target_recall": target_recall,
        "matched": matched,
    }
    if matched is not None:
        store.set_search_param(int(matched["ef"]))
        out["qps_by_clients"] = {c: concurrent_qps(store, queries, k, c) for c in clients}
        _, flat = _timed_queries(store, queries[:200], k, {"source": "s1"})
        out["filtered"] = _percentiles(flat)
        probe = np.asarray(rng.standard_normal(base.shape[1]), dtype=np.float32)
        out["write_to_searchable_s"] = wait_until_searchable(
            store, "_probe_", probe / np.linalg.norm(probe)
        )
    out["rss_delta_mb"] = (proc.memory_info().rss - rss0) / 2**20
    out["container_memory_mb"] = docker_memory_mb(container)
    store.close()
    return out


def bench_graph_store(
    store: GraphStore,
    nodes: list[Node],
    edges: list[Edge],
    *,
    n_entities: int = 1000,
    seed: int = 13,
    container: str | None = None,
) -> dict[str, Any]:
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    store.clear()
    t0 = time.perf_counter()
    store.bulk_load(nodes, edges)
    load_s = time.perf_counter() - t0
    names = [n["id"] for n in nodes]
    sample = random.Random(seed).sample(names, min(n_entities, len(names)))
    out: dict[str, Any] = {
        "backend": store.backend,
        "nodes": len(nodes),
        "edges": len(edges),
        "bulk_load_s": load_s,
        "counts_after_load": store.counts(),
    }
    for label, fn in (
        ("lookup", lambda n: store.get_entity(n)),
        ("hop1", lambda n: store.neighbors(n, 1)),
        ("hop2", lambda n: store.neighbors(n, 2)),
    ):
        reps = []
        for _ in range(3):
            lat = []
            for n in sample:
                t = time.perf_counter()
                fn(n)
                lat.append(time.perf_counter() - t)
            reps.append(lat)
        best = min(reps, key=lambda lat: statistics.median(lat))
        out[label] = _percentiles(best)
    out["rss_delta_mb"] = (proc.memory_info().rss - rss0) / 2**20
    out["container_memory_mb"] = docker_memory_mb(container)
    return out
