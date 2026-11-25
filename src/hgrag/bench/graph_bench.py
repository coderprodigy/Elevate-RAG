"""E4 Level A graph benchmark: bulk load, lookup, 1-hop and 2-hop latency, memory.

Every store's answers are checked against NetworkX (the reference), so a fast-but-wrong store
cannot win: ``agreement`` is the share of sampled neighbourhoods identical to NetworkX's.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hgrag.bench.graph_stores import Edge, GraphStore, NetworkXStore
from hgrag.eval.metrics.efficiency import percentile


@dataclass
class GraphStoreReport:
    store: str
    nodes: int
    edges: int
    load_seconds: float
    lookup_ms: dict[str, float]
    one_hop_ms: dict[str, float]
    two_hop_ms: dict[str, float]
    agreement: float  # share of 1- and 2-hop answers identical to NetworkX
    memory_mb: float | None


def _latencies(fn: Callable[[str], object], names: Sequence[str]) -> dict[str, float]:
    samples = []
    for name in names:
        started = time.perf_counter()
        fn(name)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50": round(percentile(samples, 50), 3),
        "p95": round(percentile(samples, 95), 3),
        "mean": round(statistics.fmean(samples), 3),
    }


def benchmark_graph_store(
    store: GraphStore,
    nodes: Sequence[str],
    edges: Sequence[Edge],
    *,
    n_entities: int = 1000,
    seed: int = 750,
    memory_probe: Callable[[], float | None] = lambda: None,
) -> GraphStoreReport:
    reference = NetworkXStore()
    reference.load(nodes, edges)
    sample = random.Random(seed).sample(list(nodes), min(n_entities, len(nodes)))

    started = time.perf_counter()
    store.load(nodes, edges)
    load_seconds = time.perf_counter() - started

    for name in sample[:50]:  # warm-up
        store.neighbours(name, 1)

    agree = sum(
        store.neighbours(name, hops) == reference.neighbours(name, hops)
        for name in sample[:200]
        for hops in (1, 2)
    )
    return GraphStoreReport(
        store=store.name,
        nodes=len(nodes),
        edges=len(edges),
        load_seconds=round(load_seconds, 2),
        lookup_ms=_latencies(store.lookup, sample),
        one_hop_ms=_latencies(lambda n: store.neighbours(n, 1), sample),
        two_hop_ms=_latencies(lambda n: store.neighbours(n, 2), sample),
        agreement=round(agree / max(2 * min(200, len(sample)), 1), 4),
        memory_mb=memory_probe(),
    )
