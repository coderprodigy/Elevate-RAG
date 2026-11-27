from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from elevate_rag.benchmarks.dbbench import (
    bench_graph_store,
    bench_vector_store,
    exact_ground_truth,
    recall_at,
)
from elevate_rag.config import VectorStoreConfig
from elevate_rag.evaluation import llm_metrics
from elevate_rag.reporting import report as R
from elevate_rag.storage.graph import NetworkXStore
from elevate_rag.storage.vector import ExactStore, make_vector_store
from elevate_rag.utils import _JsonFormatter


def _vecs(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_vector_bench_on_exact_store_reaches_matched_recall() -> None:
    base, queries = _vecs(300), _vecs(40, seed=1)
    ids = [f"v{i}" for i in range(300)]
    out = bench_vector_store(
        lambda: ExactStore(VectorStoreConfig(), "b", 32),
        ids,
        base,
        queries,
        ef_values=(16, 64),
        clients=(1, 4),
        warmup=20,
    )
    assert out["matched"]["recall@10"] == 1.0  # exact search is the ground truth
    assert set(out["qps_by_clients"]) == {1, 4} and out["write_to_searchable_s"] >= 0
    assert {"p50_ms", "p95_ms", "p99_ms"} <= set(out["filtered"])


def test_faiss_hnsw_store_matches_exact_at_high_ef() -> None:
    pytest.importorskip("faiss")
    base, queries = _vecs(500), _vecs(30, seed=2)
    ids = [f"v{i}" for i in range(500)]
    store = make_vector_store(VectorStoreConfig(backend="faiss", hnsw_ef_search=256), "f", 32)
    store.add(ids, base, [{"source": "a" if i % 2 else "b"} for i in range(500)])
    truth = [[ids[i] for i in row] for row in exact_ground_truth(base, queries, 10)]
    got = [[h[0] for h in store.search(q, 10)] for q in queries]
    assert recall_at(got, truth, 10) >= 0.95
    filtered = store.search(queries[0], 5, {"source": "a"})
    assert filtered and all(int(h[0][1:]) % 2 == 1 for h in filtered)
    out = bench_vector_store(
        lambda: make_vector_store(VectorStoreConfig(backend="faiss"), "f2", 32),
        ids,
        base,
        queries,
        ef_values=(8, 64, 256),
        clients=(1, 2),
        warmup=10,
    )
    assert out["matched"] is not None and out["curve"][-1]["recall@10"] >= 0.95


def test_graph_bench_networkx() -> None:
    nodes = [{"id": f"n{i}"} for i in range(50)]
    edges = [{"src": f"n{i}", "tgt": f"n{(i + 1) % 50}"} for i in range(50)]
    out = bench_graph_store(NetworkXStore(), nodes, edges, n_entities=20)
    assert out["counts_after_load"] == (50, 50)
    assert {"lookup", "hop1", "hop2"} <= set(out) and out["container_memory_mb"] is None


# ------------------------------------------------------------------------------- LLM metrics


async def test_llm_metrics_follow_ragas_definitions(scripted) -> None:
    llm = scripted(
        [
            '{"claims": [{"claim": "a", "supported": 1}, {"claim": "b", "supported": 0}]}',
            '{"statements": [{"statement": "x", "attributed": 1}]}',
            '{"useful": [0, 1, 1]}',
            "not json",
        ]
    )
    assert await llm_metrics.faithfulness(llm, "q", "a and b", ["ctx"]) == 0.5
    assert await llm_metrics.context_recall(llm, "q", "x", ["ctx"]) == 1.0
    # relevant at ranks 2 and 3: (1/2 + 2/3) / 2
    assert await llm_metrics.context_precision(llm, "q", "x", ["a", "b", "c"]) == pytest.approx(
        (0.5 + 2 / 3) / 2
    )
    assert await llm_metrics.faithfulness(llm, "q", "a", ["ctx"]) is None  # unparseable
    assert await llm_metrics.faithfulness(llm, "q", "", ["ctx"]) is None  # undefined
    assert await llm_metrics.context_recall(llm, "q", "x", []) == 0.0
    assert await llm_metrics.context_precision(llm, "q", "", ["a"]) is None


# ------------------------------------------------------------------------------- reporting


def _fake_runs(tmp_path: Path) -> list[dict]:
    runs = []
    for i, (name, mode, gen) in enumerate(
        [("V4", "v4", "g-small"), ("local", "local", "g-small"), ("V4b", "v4", "g-large")]
    ):
        d = tmp_path / name
        d.mkdir()
        rows = [
            {"question_id": f"q{j}", "scores": {"correct": (j + i) % 2 == 0}} for j in range(20)
        ]
        (d / "scored.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        acc = {"mean": 0.5, "ci_low": 0.3, "ci_high": 0.7, "n": 20}
        runs.append(
            {
                "dir": d,
                "manifest": {
                    "demo": False,
                    "definition": {
                        "experiment": "E",
                        "dataset": "ds",
                        "variant": {"name": name, "mode": mode},
                        "config": {"models": {"generator": {"model": gen}}},
                    },
                },
                "summary": {
                    "overall": {"n": 20, "correct": acc, "f1": acc},
                    "by_query_type": {
                        "single_hop": {"n": 10, "correct": acc},
                        "multi_hop": {"n": 10, "correct": None},
                    },
                    "efficiency": {"prompt_tokens_mean": 100.0 * (i + 1)},
                    "error_rate": 0.0,
                },
            }
        )
    return runs


def test_report_tables_and_all_figures(tmp_path: Path) -> None:
    runs = _fake_runs(tmp_path)
    table = R.results_table(runs)
    assert table[0]["accuracy"] == 0.5 and table[0]["accuracy_ci"] == "[0.300, 0.700]"
    R.write_csv(tmp_path / "t.csv", table, R.TABLE_FIELDS)
    comps = R.paired_comparisons(runs, "V4")
    assert len(comps) == 2 and comps[0]["n"] == 20
    out = tmp_path / "fig"
    paths = [
        R.fig_accuracy_by_type(runs, "ds", out / "a.png"),
        R.fig_router_scatter(runs, "ds", out / "b.png"),
        R.fig_factor_importance(comps, out / "c.png"),
        R.fig_interaction(runs, out / "d.png"),
        R.fig_recall_qps(
            [
                {
                    "backend": "exact",
                    "curve": [
                        {"recall@10": 0.9, "qps_1_client": 100},
                        {"recall@10": 1.0, "qps_1_client": 50},
                    ],
                }
            ],
            out / "f.png",
        ),
    ]
    assert all(p is not None and p.stat().st_size > 1000 for p in paths)
    assert R.fig_accuracy_by_type(runs, "other", out / "x.png") is None
    assert R.fig_recall_qps([], out / "x.png") is None
    R.write_markdown_report(tmp_path / "REPORT.md", table, comps, [p for p in paths if p])
    assert "Paired comparisons" in (tmp_path / "REPORT.md").read_text()


def test_json_log_formatter_redacts_secrets() -> None:
    rec = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "key sk-abcdefghijklmnop used", None, None
    )
    out = json.loads(_JsonFormatter().format(rec))
    assert "sk-abc" not in out["msg"] and "<redacted>" in out["msg"]
