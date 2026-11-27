import json
from pathlib import Path

import pytest

from elevate_rag.cli import main
from elevate_rag.data import read_jsonl, write_jsonl
from elevate_rag.interop import export_reference, import_reference

ROOT = Path(__file__).resolve().parents[2]


def test_cli_runs_stages_and_validates_manifest(tmp_path, data_path, capsys):
    pytest.importorskip("faiss")
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    args = [
        "--config",
        str(ROOT / "configs" / "synthetic.toml"),
        "--artifacts",
        str(artifacts),
        "--results",
        str(results),
    ]
    dataset = str(data_path)
    query = "Which beacon belongs to Station Aster?"
    assert main([*args, "validate-data", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["synthetic"] is True
    assert main([*args, "run", dataset, "--mode", "V1", "--split", "dev", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["required_modes"] == ["V1"]
    assert not results.exists()
    assert main([*args, "retrieve", dataset, query, "--mode", "V3", "--rerank"]) == 0
    hits = json.loads(capsys.readouterr().out)["hits"]
    assert hits and all(hit["rerank_score"] is None for hit in hits)
    assert main([*args, "answer", dataset, query, "--mode", "V4"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["answer"]["text"] == "Beacon Aster [1]"
    assert answer["context"]["passages"][0]["doc_id"] == "fixture-d00"
    assert main([*args, "index", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == 20
    assert list(artifacts.glob("vectors/*/index.faiss"))
    assert (
        main([*args, "run", dataset, "--mode", "V4", "--run-id", "cli-v4", "--split", "dev"]) == 0
    )
    run = Path(json.loads(capsys.readouterr().out)["run"])
    assert main(["evaluate", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["attempted"] == 4


def test_level_a_faiss_smoke(tmp_path, data_path, capsys):
    pytest.importorskip("faiss")
    out = tmp_path / "benchmark"
    assert (
        main(
            [
                "--config",
                str(ROOT / "configs" / "synthetic.toml"),
                "--artifacts",
                str(tmp_path / "indexes"),
                "benchmark-vector",
                str(data_path),
                str(out),
                "--backend",
                "faiss",
                "--limit",
                "2",
                "--ef",
                "32",
                "--warmup",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["matched_target"]["recall_at_10"] >= 0.95
    assert result["write_to_searchable_seconds"] is not None
    assert (out / "vector-recall-qps.csv").is_file()


def test_external_comparison_preserves_boundary_and_failures(tmp_path, data_path):
    exported = tmp_path / "export"
    manifest = export_reference(data_path, exported, "dev", framework="hipporag2", limit=2, seed=42)
    queries = list(read_jsonl(exported / "queries.jsonl"))
    assert len(queries) == 2 and set(queries[0]) == {"id", "question"}
    raw = tmp_path / "external.jsonl"
    outcomes = [
        {"id": manifest["qids"][0], "status": "ok", "answer": "Example answer", "contexts": []},
        {"id": manifest["qids"][1], "status": "error", "contexts": []},
    ]
    write_jsonl(raw, outcomes)
    out = tmp_path / "isolated"
    result = import_reference(data_path, exported, raw, "local-test", out)
    assert result["isolated_external_run"]
    assert [row["status"] for row in read_jsonl(out / "questions.jsonl")] == ["ok", "error"]
    write_jsonl(raw, outcomes[:1])
    with pytest.raises(ValueError, match="aligned"):
        import_reference(data_path, exported, raw, "local-test", out)
