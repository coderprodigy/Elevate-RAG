"""Integration with the real LightRAG 1.5.7 package (offline providers, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elevate_rag.config import PipelineConfig
from elevate_rag.data.prepare import DatasetStore
from elevate_rag.graph.lightrag_engine import GRAPH_MODES, LightRAGEngine
from elevate_rag.pipeline import ElevateRAG
from elevate_rag.types import Query
from elevate_rag.utils import ElevateError

pytestmark = pytest.mark.integration


async def _built(cfg: PipelineConfig) -> ElevateRAG:
    ds = DatasetStore.open(cfg.data_dir, "synthetic")
    s = ElevateRAG(cfg)
    await s.build_index(ds.documents(), dataset_version=ds.version)
    return s


async def test_chunk_identity_and_provenance(tmp_cfg: PipelineConfig) -> None:
    s = await _built(tmp_cfg)
    try:
        assert s.store is not None and s.graph is not None
        # D-02 regression guard: LightRAG's stored chunk IDs are reproducible from our chunks.
        rows = await s.graph.rag.text_chunks.get_by_ids(
            [LightRAGEngine.lightrag_chunk_id(c) for c in s.store.chunks()]
        )
        assert all(r is not None for r in rows)
        assert [r["content"] for r in rows] == [c.text for c in s.store.chunks()]
        nodes, edges = await s.graph.export_graph()
        assert nodes and edges
        for item in nodes + edges:  # FR-G1: every entity/relation traces back to our chunks
            assert item["chunk_ids"], item
            assert all(cid in s.store for cid in item["chunk_ids"])
    finally:
        await s.close()


async def test_every_mode_retrieves_mapped_items(tmp_cfg: PipelineConfig) -> None:
    s = await _built(tmp_cfg)
    try:
        assert s.store is not None
        q = Query("q", "Who founded the company Lumenward?")
        for mode in GRAPH_MODES:
            res = await s.retrieve(q, mode)
            for it in res.items:
                if it.kind == "chunk":
                    assert it.item_id in s.store
                else:
                    assert all(c in s.store for c in it.source_chunk_ids)
        local = await s.retrieve(q, "local")
        assert any(it.kind == "entity" for it in local.items)
        naive = await s.retrieve(q, "naive")
        assert all(it.kind == "chunk" for it in naive.items)
        assert naive.metadata.get("reranked")  # shared reranker applies to graph modes
        with pytest.raises(Exception, match="unknown mode"):
            await s.retrieve(q, "bogus")
    finally:
        await s.close()


async def test_two_hop_expansion_adds_relations(tmp_path: Path) -> None:
    from elevate_rag.data.synthetic import write_synthetic
    from tests.conftest import offline_config

    cfg1 = offline_config(tmp_path)
    write_synthetic(Path(cfg1.data_dir))
    s = await _built(cfg1)
    q = Query("q", "Who founded Velmora?")
    one = await s.retrieve(q, "local")
    await s.close()
    cfg2 = cfg1.model_copy(
        update={"retrieval": cfg1.retrieval.model_copy(update={"graph_depth": 2})}
    )
    assert cfg2.index_key() == cfg1.index_key()  # depth is a query-time knob: same index
    s2 = await ElevateRAG(cfg2).open()
    try:
        two = await s2.retrieve(q, "local")
        hop2 = [it for it in two.items if it.metadata.get("hop") == 2]
        assert len(two.items) >= len(one.items) and hop2
        assert all(it.kind == "relation" for it in hop2)
    finally:
        await s2.close()


async def test_entity_merge_through_lightrag_api(tmp_cfg: PipelineConfig) -> None:
    s = await _built(tmp_cfg)
    try:
        assert s.graph is not None
        nodes, _ = await s.graph.export_graph()
        names = sorted(n["id"] for n in nodes)
        a, b = names[0], names[1]
        await s.graph.merge_entities([b], a)
        after, edges = await s.graph.export_graph()
        ids = {n["id"] for n in after}
        assert a in ids and b not in ids and len(after) == len(nodes) - 1
        assert all(e["source"] != b and e["target"] != b for e in edges)
    finally:
        await s.close()


async def test_index_resume_skips_finished_documents(
    tmp_cfg: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = await _built(tmp_cfg)
    await s.close()
    progress = s.index_path / "lightrag_progress.txt"
    assert len(progress.read_text().split()) == 12
    calls: list[Any] = []
    s2 = ElevateRAG(tmp_cfg)
    engine = await s2._open_graph()
    monkeypatch.setattr(
        engine.rag, "ainsert_custom_chunks", lambda *a, **k: calls.append(a)
    )  # would explode if awaited
    ds = DatasetStore.open(tmp_cfg.data_dir, "synthetic")
    await s2.build_index(ds.documents())
    await s2.close()
    assert calls == []  # every document was already done


async def test_malformed_extraction_output_does_not_break_indexing(tmp_cfg: PipelineConfig) -> None:
    from elevate_rag.providers.llm import LLMResult, OfflineLLM

    class Garbage(OfflineLLM):
        async def _call(self, messages: Any, system: Any, max_tokens: int, hints: Any) -> LLMResult:
            return LLMResult(
                text="### not the expected format ###\n{{{",
                model="garbage",
                prompt_tokens=1,
                completion_tokens=1,
                latency_s=0.0,
            )

    s = ElevateRAG(tmp_cfg)
    assert s.extractor is not None
    s.extractor = Garbage(s.extractor.cfg, role="extractor", ledger=s.ledger)
    ds = DatasetStore.open(tmp_cfg.data_dir, "synthetic")
    m = await s.build_index(ds.documents())
    try:
        assert m["graph"]["failed"] == {} and m["graph"]["built"]
        nodes, _ = await s.graph.export_graph()  # type: ignore[union-attr]
        assert nodes == []  # nothing extracted, nothing invented
        rec = await s.answer(Query("q", "Who founded Lumenward?"), "local")
        assert rec.error is None  # empty graph context is a legitimate empty retrieval
    finally:
        await s.close()


async def test_open_without_graph_fails_for_graph_modes(tmp_cfg: PipelineConfig) -> None:
    ds = DatasetStore.open(tmp_cfg.data_dir, "synthetic")
    s = ElevateRAG(tmp_cfg)
    await s.build_index(ds.documents(), graph=False)
    rec = await s.answer(Query("q", "x"), "local")
    assert rec.error and "graph" in rec.error  # recorded per-question, not swallowed
    await s.close()
    with pytest.raises(ElevateError, match="graph index requested"):
        await ElevateRAG(tmp_cfg).open(graph=True)
    vec = await ElevateRAG(tmp_cfg).open(graph=False)
    assert (await vec.answer(Query("q", "When was Velmora founded?"), "v4")).error is None
    await vec.close()


async def test_two_indexes_in_one_process_are_isolated(tmp_path: Path) -> None:
    """LightRAG shares state per workspace inside a process; each index must get its own."""
    from elevate_rag.data.synthetic import write_synthetic
    from tests.conftest import offline_config

    a = offline_config(tmp_path / "a")
    b = offline_config(tmp_path / "b", chunking={"chunk_tokens": 64, "overlap_tokens": 8})
    for c in (a, b):
        write_synthetic(Path(c.data_dir))
    assert a.index_key() != b.index_key()
    sa = await _built(a)
    na, _ = await sa.graph.export_graph()  # type: ignore[union-attr]
    await sa.close()
    assert na
    sb = await _built(b)
    try:
        m = json_load(sb.index_path / "index_manifest.json")
        assert m["graph"]["failed"] == {} and m["graph"]["built"]
        nb, _ = await sb.graph.export_graph()  # type: ignore[union-attr]
        assert nb  # b really extracted its own graph instead of seeing a's documents
        rows = await sb.graph.rag.text_chunks.get_by_ids(  # type: ignore[union-attr]
            [LightRAGEngine.lightrag_chunk_id(c) for c in sb.store.chunks()]
        )  # type: ignore[union-attr]
        assert all(r is not None for r in rows)
    finally:
        await sb.close()


def json_load(path: Path) -> Any:
    import json

    return json.loads(path.read_text())


async def test_community_layer_ablation(tmp_path: Path) -> None:
    from elevate_rag.data.synthetic import write_synthetic
    from tests.conftest import offline_config

    cfg = offline_config(tmp_path, community={"enabled": True, "min_size": 2, "top_reports": 2})
    write_synthetic(Path(cfg.data_dir))
    s = await _built(cfg)
    try:
        m = json_load(s.index_path / "index_manifest.json")
        assert m["graph"]["communities"]["reports"] >= 1
        res = await s.retrieve(Query("q", "What are the main themes of Ostrand?"), "global")
        comm = [it for it in res.items if it.kind == "community"]
        assert 1 <= len(comm) <= 2 and all(
            c in s.store
            for it in comm  # type: ignore[operator]
            for c in it.source_chunk_ids
        )
    finally:
        await s.close()
    reopened = await ElevateRAG(cfg).open()
    assert reopened.communities is not None and reopened.communities.reports
    await reopened.close()
