import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from elevate_rag.graph import build_graph, graph_key
from elevate_rag.pipeline import index_parts, open_pipeline
from elevate_rag.providers import Cache, Providers
from elevate_rag.suite import Arm, execute_suite, prepare_arms

ROOT = Path(__file__).resolve().parents[2]


def test_suite_resolves_one_factor_and_keeps_independent_workflows(tmp_path, settings):
    source = tmp_path / "spec.toml"
    source.write_text("""
[datasets]
multihop-rag = "multihop"
[models.reranking.none]
provider = "disabled"
model = "none"
revision = "1"
""")
    arms, separate = prepare_arms(source, "E2", settings)
    assert len(arms) == 5 and len(separate) == 4
    assert all(arm.settings == settings for arm in arms)
    model, missing = prepare_arms(source, "E5", settings)
    assert len(model) == 1 and model[0].case["variant"] == "none"
    assert model[0].settings.models["reranking"].provider == "disabled"
    assert missing and model[0].settings.models["embedding"] == settings.models["embedding"]
    database, micro = prepare_arms(source, "E4", settings)
    assert len(micro) == 8
    qdrant = next(arm for arm in database if arm.case["variant"] == "qdrant")
    assert qdrant.settings.vector_store == qdrant.settings.graph_vector_store == "qdrant"
    assert (qdrant.limit, qdrant.warmup, qdrant.repeats) == (300, 1000, 3)
    routers, skipped = prepare_arms(source, "E3", settings)
    assert any(arm.router == "R4" for arm in routers)
    assert {"R2-query-type", "R2-outcome", "offline-oracle"} <= {row["variant"] for row in skipped}
    ablations, _ = prepare_arms(source, "ablations", settings)
    assert {arm.settings.graph_depth for arm in ablations} == {1, 2}
    assert {arm.settings.community for arm in ablations} == {True, False}


def test_suite_runs_repetitions_and_summarizes(tmp_path, data_path, settings):
    arm = Arm(
        {"block": "E1", "dataset": "synthetic", "variant": "V1", "factor": "retrieval"},
        data_path,
        settings,
        "V1",
        repeats=2,
        warmup=1,
        limit=2,
    )
    opts = dict(
        split="dev", root=tmp_path / "artifacts", results=tmp_path / "results", retry_failed=False
    )
    [planned] = asyncio.run(execute_suite([arm], dry_run=True, **opts))
    assert planned["questions"] == 2 and planned["repetitions"] == 2
    assert "paths" not in planned
    [completed] = asyncio.run(execute_suite([arm], dry_run=False, **opts))
    assert len(completed["paths"]) == 2
    assert completed["median"]["correct"] == 1
    assert all((Path(path) / "warmup.json").exists() for path in completed["paths"])


def test_graph_pilot_does_not_publish_full_index(tmp_path, data_path, settings):
    pytest.importorskip("lightrag")
    pytest.importorskip("faiss")
    settings = replace(settings, graph_depth=2)

    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            catalog, chunks_key, _ = index_parts(data_path, tmp_path, providers)
            identity = graph_key(settings, chunks_key)
            path = tmp_path / "graphs" / identity
            build_args = (
                identity,
                list(catalog.values()),
                providers,
                providers.tokenizer(context=True),
            )
            kwargs = dict(root=tmp_path, chunks_key=chunks_key)
            pilot = await build_graph(path / "pilot", *build_args, pilot=True, **kwargs)
            assert pilot["chunks"] == 1 and pilot["corpus_chunks"] == 20
            assert pilot["projected_full_usd"] == 0
            with pytest.raises(ValueError, match="missing or incomplete"):
                async with open_pipeline(data_path, tmp_path, providers, ["mix"]):
                    pytest.fail("pilot accepted as a full graph")
            assert len(list((tmp_path / "vectors").glob("*/index.faiss"))) == 1
            full = await build_graph(path / "full", *build_args, pilot=False, **kwargs)
            assert full["passage_index_key"] != pilot["passage_index_key"]
            async with open_pipeline(data_path, tmp_path, providers, ["mix"]) as pipeline:
                hits, trace = await pipeline.retrieve(
                    "Which beacon belongs to Station Aster?", "mix"
                )
                assert hits and trace["relations"]

    asyncio.run(check())
