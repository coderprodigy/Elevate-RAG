from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elevate_rag.config import GraphStoreConfig, ProviderConfig, VectorStoreConfig
from elevate_rag.evaluation.harness import summarize
from elevate_rag.experiments.runner import ExperimentSpec, Variant, estimate, expand, validate
from elevate_rag.graph.quality import answer_entity_coverage, graph_shape
from elevate_rag.graph.resolution import normalize_entity, plan_merges
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.providers.base import UsageLedger
from elevate_rag.providers.embeddings import HashEmbedder
from elevate_rag.storage.graph import NetworkXStore, make_graph_store
from elevate_rag.storage.vector import ExactStore, make_vector_store, wait_until_searchable
from elevate_rag.types import Chunk, Question
from elevate_rag.utils import ConfigError, StorageError

# ------------------------------------------------------------------------------- resolution


def test_normalize_entity() -> None:
    assert normalize_entity("The  Royal-Survey Academy") == "royal survey academy"


def test_entity_key_ignores_spacing_and_camel_case() -> None:
    from elevate_rag.graph.resolution import entity_key

    same = ["UnitedKingdom", "United Kingdom", "the united kingdom"]
    assert len({entity_key(x) for x in same}) == 1
    assert entity_key("TheClickFive") == entity_key("The Click Five")  # article still stripped
    assert entity_key("iPhone") == entity_key("iphone")
    assert entity_key("2004FrenchOpen") == entity_key("2004 French Open")
    assert entity_key("Red Sea") != entity_key("Dead Sea")
    plan = plan_merges([{"id": "JorgeAmado"}, {"id": "Jorge Amado"}], None, 1.0)
    assert plan.n_merged == 1  # observed in the 7B G1 graph: 34 such split entities


def test_plan_merges_by_name_and_embedding_with_type_guard() -> None:
    nodes = [
        {"id": "Barack Obama", "entity_type": "person", "source_id": "c1<SEP>c2"},
        {"id": "barack obama", "entity_type": "person", "source_id": "c3"},
        {"id": "Paris", "entity_type": "city"},
        {"id": "paris", "entity_type": "person"},
        {"id": "Velmora", "entity_type": "location"},
    ]
    plan = plan_merges(nodes, None, 1.0)
    groups = {g["target"]: g["sources"] for g in plan.groups}
    assert groups["Barack Obama"] == ["barack obama"]  # higher-degree name wins
    assert plan.n_merged == 2  # name normalisation merges Paris/paris regardless of type
    emb = HashEmbedder(
        ProviderConfig(provider="hash", model="h", dimension=256), ledger=UsageLedger()
    )
    nodes2 = [
        {"id": "Ostrand Tidal Act", "entity_type": "law"},
        {"id": "Ostrand Tidal Act 2015", "entity_type": "law"},
        {"id": "Ostrand Tidal Act of", "entity_type": "event"},
    ]
    p2 = plan_merges(nodes2, emb, 0.6)
    merged = {s for g in p2.groups for s in [g["target"], *g["sources"]]}
    assert "Ostrand Tidal Act 2015" in merged
    assert "Ostrand Tidal Act of" not in merged  # different entity type → never merged
    assert all(
        link["by"] in ("embedding", "normalized_name") for g in p2.groups for link in g["links"]
    )
    assert plan_merges(nodes2, emb, 0.6, max_merges=0).groups == []


# ------------------------------------------------------------------------------- quality


def test_answer_entity_coverage_rules() -> None:
    qs = [
        Question(question_id="1", dataset="d", question="?", answers=["Obama"]),
        Question(question_id="2", dataset="d", question="?", answers=["Kessering"]),
        Question(question_id="3", dataset="d", question="?", answers=["yes"]),
        Question(
            question_id="4",
            dataset="d",
            question="?",
            answers=["the Royal Academy"],
            query_type="multi_hop",
        ),
        Question(question_id="5", dataset="d", question="?", answers=[], answerable=False),
    ]
    cov = answer_entity_coverage(qs, ["Barack Obama", "Royal Academy", "Velmora"])
    assert cov["eligible"] == 3 and cov["excluded_non_entity"] == 1
    assert cov["covered"] == 2 and cov["coverage"] == pytest.approx(2 / 3)
    assert cov["by_query_type"]["multi_hop"] == 1.0 and "Kessering" in cov["missed_examples"]
    joined = answer_entity_coverage(qs[:2], ["BarackObama", "KesseringBridge"])
    assert joined["covered"] == 2  # space-stripped node names still count


def _store() -> ChunkStore:
    return ChunkStore(
        [
            Chunk(
                chunk_id="d::c0000",
                doc_id="d",
                index=0,
                text="A founded B.",
                n_tokens=3,
                start_char=0,
                end_char=12,
            )
        ]
    )


def test_graph_shape_and_provenance_counts() -> None:
    nodes = [
        {"id": "A", "chunk_ids": ["d::c0000"]},
        {"id": "a", "chunk_ids": ["d::c0000"]},
        {"id": "B", "chunk_ids": []},
        {"id": "C", "chunk_ids": ["d::c0000"]},
    ]
    edges = [
        {"source": "A", "target": "B", "chunk_ids": ["d::c0000"]},
        {"source": "B", "target": "C", "chunk_ids": []},
    ]
    s = graph_shape(nodes, edges, _store())
    assert (s["nodes"], s["edges"]) == (4, 2) and s["avg_degree"] == 1.0
    assert s["duplicate_rate"] == 0.5 and s["isolated_nodes"] == 1
    assert s["nodes_without_provenance"] == 1 and s["edges_without_provenance"] == 1


# ------------------------------------------------------------------------------- storage


def test_exact_store_search_filter_persist(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 16)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"c{i:02d}" for i in range(50)]
    s = ExactStore(VectorStoreConfig(), "t", 16)
    s.add(ids, vecs, [{"source": "a" if i % 2 else "b"} for i in range(50)])
    assert s.search(vecs[7], 3)[0][0] == "c07"
    assert all(int(h[0][1:]) % 2 == 1 for h in s.search(vecs[7], 5, {"source": "a"}))
    s.save(tmp_path)
    again = ExactStore.load(VectorStoreConfig(), "t", tmp_path)
    assert again.search(vecs[3], 1) == s.search(vecs[3], 1) and again.count() == 50
    assert wait_until_searchable(s, "probe", vecs[0]) >= 0
    with pytest.raises(StorageError):
        s.add(["x"], np.ones((1, 8)))
    assert ExactStore(VectorStoreConfig(), "e", 4).search(np.ones(4), 3) == []


def test_server_stores_fail_loudly_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ELEVATE_QDRANT_URL", "ELEVATE_MILVUS_URI", "ELEVATE_PG_DSN"):
        monkeypatch.delenv(var, raising=False)
    for backend in ("qdrant", "milvus", "pgvector"):
        with pytest.raises(ConfigError):  # missing extra or missing URL — never a fallback
            make_vector_store(VectorStoreConfig(backend=backend), "t", 4)
    monkeypatch.delenv("ELEVATE_NEO4J_URI", raising=False)
    with pytest.raises(ConfigError):
        make_graph_store(GraphStoreConfig(backend="neo4j"))


def test_networkx_store_hops() -> None:
    g = NetworkXStore()
    g.bulk_load(
        [{"id": x} for x in "ABCD"],
        [{"src": "A", "tgt": "B"}, {"src": "B", "tgt": "C"}, {"src": "C", "tgt": "D"}],
    )
    assert g.neighbors("A", 1) == {"B"} and g.neighbors("A", 2) == {"B", "C"}
    assert g.get_entity("Z") is None and g.counts() == (4, 3)
    g.clear()
    assert g.counts() == (0, 0)


# ------------------------------------------------------------------------------- experiments


def _spec(**kw: object) -> ExperimentSpec:
    base = {
        "id": "E1",
        "question": "RQ1",
        "pipeline": "pipe.yaml",
        "datasets": ["synthetic"],
        "variants": [Variant(name="V1", mode="v1"), Variant(name="local", mode="local")],
    }
    base.update(kw)
    return ExperimentSpec.model_validate(base)


def _write_pipe(tmp_path: Path, judge_family: str = "other", extra: str = "") -> None:
    (tmp_path / "pipe.yaml").write_text(f"""
name: p
models:
  embedder: {{provider: hash, model: h, dimension: 64}}
  extractor: {{provider: offline, model: x}}
  generator: {{provider: offline, model: g, family: gen}}
  judge: {{provider: offline, model: j, family: {judge_family}}}
data_dir: {tmp_path / "data"}
{extra}
""")


def test_expand_produces_stable_ids_and_graph_needs(tmp_path: Path) -> None:
    _write_pipe(tmp_path)
    runs = expand(_spec(), tmp_path)
    assert [r.run_id for r in runs] == [r.run_id for r in expand(_spec(), tmp_path)]
    assert [r.needs_graph for r in runs] == [False, True]
    assert all(r.cfg.dataset == "synthetic" for r in runs)


def test_validate_enforces_protocol_rules(tmp_path: Path) -> None:
    _write_pipe(tmp_path, judge_family="gen")
    spec = _spec(
        variants=[
            Variant(name="bad", mode="v1", router={"level": "rules"}),
            Variant(
                name="ext",
                mode="local",
                overrides={"models.extractor": {"provider": "offline", "model": "y"}},
            ),
        ]
    )
    problems = validate(spec, expand(spec, tmp_path))
    text = "\n".join(problems)
    assert "exactly one of `mode` or `router`" in text
    assert "judge family" in text
    assert "extractor comparisons are restricted" in text
    assert "dataset not prepared" in text
    _write_pipe(tmp_path, extra="budget: {max_usd: 0}")
    paid = _spec(
        variants=[
            Variant(
                name="p",
                mode="v1",
                overrides={
                    "models.generator": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-5-5",
                        "family": "claude",
                        "price_input_per_mtok": 2.0,
                        "price_output_per_mtok": 10.0,
                    }
                },
            )
        ]
    )
    assert any(
        "budget.max_usd is 0" in p for p in validate(paid, expand(paid, tmp_path), check_data=False)
    )


def test_estimate_flags_unknown_prices(tmp_path: Path) -> None:
    _write_pipe(tmp_path)
    spec = _spec(
        variants=[
            Variant(
                name="u",
                mode="v1",
                overrides={
                    "models.generator": {
                        "provider": "openai-compatible",
                        "model": "m",
                        "base_url": "http://x",
                    }
                },
            )
        ]
    )
    est = estimate(expand(spec, tmp_path))
    assert est["is_estimate"] and est["has_unknown_prices"]


# ------------------------------------------------------------------------------- harness


def _row(
    qid: str,
    correct: bool | None,
    qtype: str = "single_hop",
    failed: bool = False,
    cost: float | None = 0.0,
) -> dict:
    return {
        "question_id": qid,
        "mode": "v4",
        "answer": "a",
        "refused": False,
        "invalid_citations": [],
        "timings": {"retrieval": 0.1, "generation": 0.2},
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "context_tokens": 50,
        "cost_usd": cost,
        "retrieved_chunk_ids": ["d1::c0000"],
        "context_item_ids": [],
        "context_items": [],
        "route": {},
        "scores": {
            "query_type": qtype,
            "answerable": True,
            "failed": failed,
            "correct": correct,
            "em": 1.0 if correct else 0.0,
            "recall@10": None,
            "judge": "invalid" if correct is None else "correct",
        },
    }


def test_summarize_counts_failures_and_undefined() -> None:
    rows = [
        _row("a", True),
        _row("b", False, "multi_hop", failed=True),
        _row("c", None),
        _row("d", True, cost=None),
    ]
    s = summarize(rows)
    assert s["overall"]["n"] == 4 and s["error_rate"] == 0.25
    assert s["overall"]["correct"]["mean"] == pytest.approx(2 / 3)
    assert s["overall"]["correct__undefined"] == 1 and s["judge_invalid"] == 1
    assert s["overall"]["recall@10"] is None and s["overall"]["recall@10__undefined"] == 4
    assert s["efficiency"]["cost_per_1k_queries_usd"] is None  # unknown, never 0
    assert set(s["by_query_type"]) == {"single_hop", "multi_hop"}
    assert summarize([]) == {"n": 0}


def test_latency_excludes_cache_served_queries() -> None:
    fresh, cached = _row("a", True), _row("b", True)
    fresh["timings"] = {"generation": 10.0}
    fresh["calls"] = [
        {"role": "generator:generate", "cached": False},
        {"role": "embed:query", "cached": True},
    ]
    cached["timings"] = {"generation": 0.0}
    cached["calls"] = [{"role": "generator:generate", "cached": True}]
    eff = summarize([fresh, cached])["efficiency"]
    assert eff["latency_p50_s"] == pytest.approx(10.0) and eff["latency_n"] == 1
    assert eff["cache_hit_rate"] == pytest.approx(0.5)
    only_cached = summarize([cached])["efficiency"]
    assert only_cached["latency_p50_s"] is None  # unknown, never a fake ~0 s


def test_milvus_unreachable_raises_storage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pymilvus")
    monkeypatch.setenv("ELEVATE_MILVUS_URI", "http://127.0.0.1:1")  # nothing listens here
    with pytest.raises(StorageError, match="Milvus unreachable"):
        make_vector_store(VectorStoreConfig(backend="milvus"), "t", 4)


def test_select_shard_partitions_runs_exactly_once() -> None:
    from elevate_rag.experiments.runner import select_shard
    from elevate_rag.utils import ConfigError

    runs = list(range(7))  # any sequence: order is what matters
    parts = [select_shard(runs, f"{k}/3") for k in range(3)]  # type: ignore[arg-type]
    assert sorted(x for p in parts for x in p) == runs and parts[0] == [0, 3, 6]
    assert select_shard(runs, None) == runs  # type: ignore[arg-type]
    for bad in ("3/3", "x", "1/0"):
        with pytest.raises(ConfigError):
            select_shard(runs, bad)  # type: ignore[arg-type]


def test_load_spec_can_retarget_pipeline_limit_and_datasets(tmp_path: Path) -> None:
    from elevate_rag.experiments.runner import expand, load_spec
    from elevate_rag.utils import ConfigError

    spec = Path("configs/experiments/E2_retrieval_modes.yaml")
    s, base = load_spec(
        spec, pipeline="configs/pipelines/demo_offline.yaml", limit=7, datasets=["hotpotqa"]
    )
    runs = expand(s, base)
    assert s.limit == 7 and {r.dataset for r in runs} == {"hotpotqa"}
    assert all(r.cfg.name == "demo-offline" for r in runs)
    ref_runs = expand(*load_spec(spec, datasets=["hotpotqa"]))
    assert {r.run_id for r in runs}.isdisjoint(r.run_id for r in ref_runs)  # no collisions
    with pytest.raises(ConfigError):
        load_spec(spec, datasets=["not-a-dataset"])


def test_second_judge_and_ragas_rules(tmp_path: Path) -> None:
    _write_pipe(tmp_path)
    base = {r.run_id: r.cfg.config_hash() for r in expand(_spec(), tmp_path)}
    # an unset judge2 must not change run ids (results from before judge2 keep resuming)
    assert all(r.cfg.models.judge2 is None for r in expand(_spec(), tmp_path))
    same = {
        r.run_id: r.cfg.config_hash()
        for r in expand(
            _spec(variants=[Variant(name="V1", mode="v1"), Variant(name="local", mode="local")]),
            tmp_path,
        )
    }
    assert base == same
    for fam, ok in (("gen", False), ("other", False), ("third", True)):
        spec = _spec(
            variants=[
                Variant(
                    name="V1",
                    mode="v1",
                    overrides={
                        "models.judge2": {"provider": "offline", "model": "k", "family": fam}
                    },
                )
            ]
        )
        runs = expand(spec, tmp_path)
        assert runs[0].cfg.config_hash() not in base.values()  # judge2 is part of the identity
        problems = validate(spec, runs, check_data=False)
        assert any("judge2 family" in p for p in problems) is not ok
    spec = _spec(ragas=True, variants=[Variant(name="V1", mode="v1")])
    assert any("ragas" in p for p in validate(spec, expand(spec, tmp_path), check_data=False))
