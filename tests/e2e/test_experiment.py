import asyncio
import json
from dataclasses import replace

import pytest

from elevate_rag.data import file_hash, read_jsonl
from elevate_rag.experiments import design, run
from elevate_rag.graph import build_graph, graph_key, inspect_graph
from elevate_rag.pipeline import index_parts, open_pipeline
from elevate_rag.providers import Cache, Providers
from elevate_rag.reporting import compare_runs


def test_twenty_questions_resume_and_compare(tmp_path, data_path, settings):
    pytest.importorskip("faiss")

    async def check():
        root = tmp_path / "indexes"
        output = tmp_path / "runs"
        main = await run(data_path, root, output, settings, split="all", run_id="v4")
        outcomes = list(read_jsonl(main / "questions.jsonl"))
        assert len(outcomes) == 20
        assert all(row["correct"] and row["status"] == "ok" for row in outcomes)
        assert sum(row["answer"]["refused"] for row in outcomes) == 3
        refreshed = await run(data_path, root, output, settings, split="all", run_id="v4")
        assert list(read_jsonl(refreshed / "questions.jsonl")) == outcomes
        with pytest.raises(ValueError, match="immutable"):
            await run(data_path, root, output, settings, split="all", run_id="v4", mode="V1")
        closed = await run(data_path, root, output, settings, split="all", run_id="v0", mode="V0")
        assert all(
            not row["context"] and not row["candidates"]
            for row in read_jsonl(closed / "questions.jsonl")
        )
        comparison = compare_runs([closed, main])
        assert comparison["comparisons"]["v4"]["mcnemar"]["right_only"] == 17
        assert json.loads((main / "summary.json").read_text())["complete"]

    asyncio.run(check())


def test_lightrag_persistence_modes_and_fallback(tmp_path, data_path, settings):
    pytest.importorskip("lightrag")
    pytest.importorskip("faiss")

    async def check():
        root = tmp_path / "indexes"
        settings_threshold = replace(settings, fallback_threshold=2)
        async with Providers(
            settings_threshold, Cache(root / "cache"), root / "budget.sqlite3"
        ) as providers:
            catalog, shared, _ = index_parts(data_path, root, providers)
            key = graph_key(settings_threshold, shared)
            directory = root / "graphs" / key / "full"
            async with open_pipeline(data_path, root, providers, ["V2"]) as dense:
                assert dense.vector is not None and dense.vector.size() == len(catalog)
            [passage_file] = list((root / "vectors").glob("*/index.faiss"))
            checksum = file_hash(passage_file)
            report = await build_graph(
                directory,
                key,
                list(catalog.values()),
                providers,
                providers.tokenizer(context=True),
                pilot=False,
                root=root,
                chunks_key=shared,
            )
            assert report["nodes"] == 40
            assert report["edges"] == 20
            assert report["passage_index_key"] == passage_file.parent.name
            assert list((root / "vectors").glob("*/index.faiss")) == [passage_file]
            assert file_hash(passage_file) == checksum
            modes = ["V4", "naive", "local", "global", "hybrid", "mix"]
            async with open_pipeline(data_path, root, providers, modes) as pipeline:
                for mode in modes:
                    hits, trace = await pipeline.retrieve(
                        "Which beacon belongs to Station Aster?", mode
                    )
                    assert hits, mode
                    assert all(hit.chunk_id in catalog for hit in hits)
                    if mode in {"local", "hybrid", "mix"}:
                        assert "Station Aster" in trace["entities"]
                    if mode == "naive":
                        assert trace["entities"] == []
                result = await pipeline.answer("Which beacon belongs to Station Aster?", level="R4")
                assert result.route.initial == "V4"
                assert result.route.final == "mix"
                assert result.route.fallback_reason == "low retrieval confidence"
                assert len(result.telemetry["paths"]) == 2
                assert result.answer.text == "Beacon Aster [1]"
                assert pipeline.graph is not None
                assert pipeline.graph.rag.chunks_vdb.vector is pipeline.vector
                native_id = next(iter(pipeline.graph.ids))
                native = await pipeline.graph.rag.chunks_vdb.get_vectors_by_ids([native_id])
                [stable_id] = pipeline.graph.ids[native_id]
                assert pipeline.vector is not None
                assert native == {native_id: pipeline.vector.get_vectors([stable_id])[stable_id]}
                nodes, edges = await pipeline.graph.snapshot()
                assert all(edge["chunk_ids"] for edge in edges)
                stats = inspect_graph(nodes, edges, [], 42, tmp_path / "audit", catalog=catalog)
                assert stats["answer_entity_coverage"] is None
        assert (tmp_path / "audit" / "triple-audit.jsonl").exists()

    asyncio.run(check())


def test_experiment_matrix():
    assert len(design("E1")) == 30
    assert {case["dataset"] for case in design("E5") if case["factor"] == "extraction"} == {
        "multihop-rag",
        "graphrag-medical",
    }
    assert all(case["isolated"] for case in design("E2") if case["factor"] == "external-framework")
    assert set(case["mode"] for case in design("E6")) == {"naive", "local", "mix"}


def test_interrupted_run_retains_completed_question_checkpoints(
    tmp_path, data_path, settings, monkeypatch
):
    pytest.importorskip("faiss")
    from elevate_rag.pipeline import Pipeline

    original = Pipeline.answer
    attempts = []

    async def interrupt(pipeline, query, **options):
        attempts.append(query)
        if len(attempts) == 3:
            raise asyncio.CancelledError
        return await original(pipeline, query, **options)

    async def check():
        root, results = tmp_path / "artifacts", tmp_path / "runs"
        monkeypatch.setattr(Pipeline, "answer", interrupt)
        with pytest.raises(asyncio.CancelledError):
            await run(
                data_path, root, results, settings, split="all", limit=5, run_id="interrupted"
            )
        folder = results / "synthetic" / "interrupted"
        partial = list(read_jsonl(folder / "questions.jsonl"))
        assert len(partial) == 2
        assert not json.loads((folder / "summary.json").read_text())["complete"]
        monkeypatch.setattr(Pipeline, "answer", original)
        await run(data_path, root, results, settings, split="all", limit=5, run_id="interrupted")
        final = list(read_jsonl(folder / "questions.jsonl"))
        assert len(final) == 5
        assert final[:2] == partial
        assert json.loads((folder / "summary.json").read_text())["complete"]

    asyncio.run(check())


def test_v4_tuning_freezes_development_only(tmp_path, data_path, settings):
    from elevate_rag.config import Settings
    from elevate_rag.data import questions_at
    from elevate_rag.tuning import tune_v4

    pytest.importorskip("faiss")
    out, cache = tmp_path / "frozen", tmp_path / "artifacts"
    options = {"sizes": (256, 512), "depths": (5, 8)}
    result = asyncio.run(tune_v4(data_path, cache, out, settings, **options))
    assert len(result["trials"]) == 4
    expected = {item.question_id for item in questions_at(data_path, "dev") if item.evidence}
    assert {row["question_id"] for row in result["trials"][0]["rows"]} == expected
    assert Settings.read(out / "v4.json").rerank_k == result["selected"]["rerank_k"]
    assert asyncio.run(tune_v4(data_path, cache, out, settings, **options)) == result
    with pytest.raises(ValueError, match="frozen"):
        asyncio.run(tune_v4(data_path, cache, out, settings, sizes=(256,), depths=(5,)))
