"""The tiny fixture corpus must stay consistent: later phases' tests depend on it."""

import json
from pathlib import Path

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_corpus"
QUERY_TYPES = {"single_hop", "multi_hop", "comparison", "temporal", "global", "unanswerable"}


def _read(name: str) -> list[dict[str, object]]:
    with (FIXTURE / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_every_gold_doc_exists_and_ids_are_unique() -> None:
    docs = _read("corpus.jsonl")
    queries = _read("queries.jsonl")
    doc_ids = [d["doc_id"] for d in docs]
    assert len(doc_ids) == len(set(doc_ids))
    assert len({q["qid"] for q in queries}) == len(queries)
    for query in queries:
        assert set(query["gold_doc_ids"]) <= set(doc_ids), query["qid"]  # type: ignore[arg-type]


def test_fixture_queries_match_the_query_schema() -> None:
    from hgrag.data.schema import Query

    for row in _read("queries.jsonl"):
        Query.model_validate(row)


def test_all_query_types_covered_and_unanswerables_have_no_gold() -> None:
    queries = _read("queries.jsonl")
    assert {q["query_type"] for q in queries} == QUERY_TYPES
    for query in queries:
        if not query["answerable"]:
            assert query["answers"] == [] and query["gold_doc_ids"] == []
