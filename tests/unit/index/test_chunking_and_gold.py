from itertools import pairwise

import pytest

from hgrag.config import ChunkerSpec
from hgrag.data.schema import Document, Query
from hgrag.index.chunking import chunk_documents, pack_sentences, split_sentences
from hgrag.index.gold import map_gold_chunks
from hgrag.index.tokens import count_tokens
from hgrag.retrieval.budget import apply_token_budget
from hgrag.retrieval.fusion import reciprocal_rank_fusion


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, title=f"T{doc_id}", text=text, source="test")


def test_split_sentences_respects_paragraphs_and_abbreviation_case() -> None:
    text = 'He left. "Then she came." It was 3 p.m. now\nNew paragraph here'
    assert split_sentences(text) == [
        "He left.",
        '"Then she came."',
        "It was 3 p.m. now",
        "New paragraph here",
    ]


def test_pack_sentences_caps_tokens_and_overlaps() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(40)]
    chunks = pack_sentences(sentences, max_tokens=30, overlap_tokens=8)
    assert all(count_tokens(c) <= 30 for c in chunks)
    assert len(chunks) > 1
    # each chunk after the first starts with the last sentence of the previous one
    for previous, current in pairwise(chunks):
        assert current.split(".")[0] in previous


def test_over_long_sentence_is_hard_split() -> None:
    chunks = pack_sentences(["word " * 300], max_tokens=50, overlap_tokens=0)
    assert all(count_tokens(c) <= 50 for c in chunks) and len(chunks) >= 6


def test_paragraph_chunker_keeps_short_docs_whole_and_splits_long_ones() -> None:
    spec = ChunkerSpec(kind="paragraph", max_tokens=40, overlap_tokens=0)
    short = _doc("d1", "A short paragraph. Two sentences.")
    long = _doc("d2", " ".join(f"Sentence {i} talks about things." for i in range(30)))
    chunks = list(chunk_documents([short, long], spec))
    assert sum(c.doc_id == "d1" for c in chunks) == 1
    assert sum(c.doc_id == "d2" for c in chunks) > 1
    assert chunks[0].indexed_text.startswith("Td1\n")
    assert [c.chunk_id for c in chunk_documents([short, long], spec)] == [
        c.chunk_id for c in chunks
    ]


def test_gold_mapping_exact_fuzzy_and_doc_level() -> None:
    spec = ChunkerSpec(kind="sentence", max_tokens=20, overlap_tokens=0)
    doc = _doc("d1", "The tower was built in 1900. " * 3 + "Its architect was Mira Halden. " * 3)
    chunks = list(chunk_documents([doc], spec))
    queries = [
        Query(
            qid="exact",
            question="?",
            answers=["x"],
            gold_doc_ids=["d1"],
            gold_evidence=["its architect was MIRA HALDEN"],
            query_type="single_hop",
            native_type="x",
        ),
        Query(
            qid="fuzzy",
            question="?",
            answers=["x"],
            gold_doc_ids=["d1"],
            gold_evidence=["The tower was build in 1900"],  # typo
            query_type="single_hop",
            native_type="x",
        ),
        Query(
            qid="miss",
            question="?",
            answers=["x"],
            gold_doc_ids=["d1"],
            gold_evidence=["completely unrelated statement about fish"],
            query_type="single_hop",
            native_type="x",
        ),
        Query(
            qid="doc_only",
            question="?",
            answers=["x"],
            gold_doc_ids=["d1"],
            query_type="single_hop",
            native_type="x",
        ),
    ]
    mapping = map_gold_chunks(queries, chunks)
    assert mapping.gold_chunks["exact"] and all(
        "architect" in next(c.text for c in chunks if c.chunk_id == cid)
        for cid in mapping.gold_chunks["exact"]
    )
    assert "fuzzy" in mapping.gold_chunks
    assert "miss" not in mapping.gold_chunks
    assert mapping.gold_chunks["doc_only"] == [c.chunk_id for c in chunks]
    assert (mapping.evidence_total, mapping.evidence_mapped) == (3, 2)


def test_rrf_hand_computed() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert [item for item, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_token_budget_keeps_rank_order_and_at_least_one() -> None:
    assert apply_token_budget(["a", "b", "c"], [400, 500, 100], 950) == ["a", "b"]
    assert apply_token_budget(["a", "b", "c"], [400, 500, 100], 1000) == ["a", "b", "c"]
    assert apply_token_budget(["huge"], [5000], 1000) == ["huge"]
