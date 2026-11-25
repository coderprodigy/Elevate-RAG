import json
from pathlib import Path

import pytest

from hgrag.data.schema import Query
from hgrag.data.splits import FrozenSplitError, assign_splits, load_or_create_splits
from hgrag.data.text import clean_text, doc_id_for, normalize_question


def test_clean_text_collapses_spaces_but_keeps_paragraphs() -> None:
    raw = "  First   line\r\nsecond\tline \n\n\n\nNew  paragraph  "
    assert clean_text(raw) == "First line\nsecond line\n\nNew paragraph"


def test_normalize_question_ignores_case_and_punctuation() -> None:
    assert normalize_question("Who is X's  SPOUSE?") == normalize_question("who is x s spouse")


def test_doc_id_is_content_based() -> None:
    assert doc_id_for("T", "text") == doc_id_for("T", "text")
    assert doc_id_for("T", "text") != doc_id_for("T", "text!")


def _queries(n_per_type: int) -> list[Query]:
    return [
        Query(qid=f"{t}-{i}", question="q", answers=["a"], query_type=t, native_type=t)
        for t in ("single_hop", "multi_hop")
        for i in range(n_per_type)
    ]


def test_splits_are_stratified_and_deterministic() -> None:
    queries = _queries(10)
    first = assign_splits(queries, dev_fraction=0.2, seed=1)
    assert first == assign_splits(queries, dev_fraction=0.2, seed=1)
    for query_type in ("single_hop", "multi_hop"):
        dev = [q for q, s in first.items() if s == "dev" and q.startswith(query_type)]
        assert len(dev) == 2
    assert first != assign_splits(queries, dev_fraction=0.2, seed=2)


def test_split_file_is_frozen(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    queries = _queries(5)
    created = load_or_create_splits(path, queries, 0.2, seed=1)
    assert json.loads(path.read_text())["sha256"]

    # a different seed must not change a frozen file
    assert load_or_create_splits(path, queries, 0.2, seed=99) == created
    with pytest.raises(FrozenSplitError, match="questions changed"):
        load_or_create_splits(path, queries[:-1], 0.2, seed=1)
