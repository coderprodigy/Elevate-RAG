"""Loader tests on miniature files that follow each source's *real* schema (verified 2026-11-26)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from elevate_rag.data import datasets as D
from elevate_rag.data.prepare import (
    DatasetStore,
    deduplicate,
    stratified_split,
    validate_references,
    write_processed,
)
from elevate_rag.data.synthetic import synthetic_load_result
from elevate_rag.types import Document, Question
from elevate_rag.utils import DataError


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_deterministic_sample_is_order_independent() -> None:
    items = [{"id": str(i)} for i in range(100)]
    a = D.deterministic_sample(items, 10, 13, key=lambda r: r["id"])
    b = D.deterministic_sample(list(reversed(items)), 10, 13, key=lambda r: r["id"])
    assert a == b and len(a) == 10
    assert D.deterministic_sample(items, None, 13, key=lambda r: r["id"]) == sorted(
        items, key=lambda r: r["id"]
    )


def test_nq_loader_gold_plus_distractors(tmp_path: Path) -> None:
    rows = [
        {
            "example_id": i,
            "question": f"q{i}?",
            "answers": [f"a{i}"],
            "text": f"passage {i}",
            "idx_gold_in_corpus": 1000 + i % 8,
        }
        for i in range(10)
    ]
    (tmp_path / "dev.json").write_text(json.dumps(rows))
    r = D.load_nq({"dev": tmp_path / "dev.json"}, {"n_questions": 3, "n_distractors": 4, "seed": 1})
    assert len(r.questions) == 3
    ids = {d.doc_id for d in r.documents}
    assert all(q.gold_doc_ids[0] in ids for q in r.questions)
    assert len(ids) <= 7  # deduplicated by idx_gold_in_corpus


def test_hotpotqa_loader_parquet(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"h{i}",
            "question": f"Q{i}?",
            "answer": "yes",
            "type": t,
            "level": lvl,
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "context": {
                "title": ["A", "B", "C"],
                "sentences": [["A is x. "], ["B is y."], ["C is z."]],
            },
        }
        for i, (t, lvl) in enumerate(
            [("bridge", "hard"), ("comparison", "hard"), ("bridge", "easy")]
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "v.parquet")
    r = D.load_hotpotqa({"validation": tmp_path / "v.parquet"}, {"n_questions": None, "seed": 1})
    assert [q.query_type for q in r.questions] == ["multi_hop", "comparison"]  # easy dropped
    assert len(r.questions[0].gold_doc_ids) == 2 and len(r.documents) == 3


def test_musique_excludes_unanswerable_twins(tmp_path: Path) -> None:
    def row(qid: str, ans: bool) -> dict:
        return {
            "id": qid,
            "question": f"{qid}?",
            "answer": "X",
            "answer_aliases": ["x"],
            "answerable": ans,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": f"T{qid}",
                    "paragraph_text": f"text {qid}",
                    "is_supporting": True,
                }
            ],
        }

    p = _jsonl(
        tmp_path / "m.jsonl",
        [row("2hop__1", True), row("2hop__1", False), row("3hop1__2", True), row("2hop__3", False)],
    )
    r = D.load_musique({"full_dev": p}, {"n_answerable": 2, "n_unanswerable": 5, "seed": 1})
    un = [q for q in r.questions if not q.answerable]
    assert [q.question_id for q in un] == ["mq-2hop__3-unans"]  # twin of 2hop__1 excluded
    assert all(not q.gold_doc_ids and not q.answers for q in un)
    hops = {q.question_id: q.metadata["hops"] for q in r.questions}
    assert hops["mq-3hop1__2"] == 3


def test_multihop_rag_types_and_null(tmp_path: Path) -> None:
    corpus = [
        {"title": "t1", "body": "article one", "url": "u1", "source": "S"},
        {"title": "t2", "body": "article two", "url": "u2", "source": "S"},
    ]
    qs = [
        {
            "query": "who?",
            "answer": "A",
            "question_type": "comparison_query",
            "evidence_list": [{"url": "u1", "fact": "fact one"}, {"url": "u2", "fact": "f2"}],
        },
        {
            "query": "none?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    (tmp_path / "c.json").write_text(json.dumps(corpus))
    (tmp_path / "q.json").write_text(json.dumps(qs))
    r = D.load_multihop_rag(
        {"corpus": tmp_path / "c.json", "queries": tmp_path / "q.json"}, {"seed": 1}
    )
    by = {q.query_type: q for q in r.questions}
    assert len(by["comparison"].gold_doc_ids) == 2 and by["comparison"].gold_evidence
    assert not by["null"].answerable and by["null"].answers == []


def test_graphrag_bench_and_ragrouter(tmp_path: Path) -> None:
    (tmp_path / "c.json").write_text(json.dumps([{"corpus_name": "Medical", "context": "Text."}]))
    (tmp_path / "q.json").write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "question": "q?",
                    "answer": "a",
                    "question_type": "Fact Retrieval",
                    "evidence": ["Text."],
                },
                {
                    "id": "m2",
                    "question": "q2?",
                    "answer": "b",
                    "question_type": "Creative Generation",
                    "evidence": [],
                },
            ]
        )
    )
    r = D.load_grb_medical(
        {"corpus": tmp_path / "c.json", "questions": tmp_path / "q.json"}, {"seed": 1}
    )
    assert {q.query_type for q in r.questions} == {"single_hop", "creative"}
    assert r.questions[0].metadata["level"] == 1
    rr = _jsonl(
        tmp_path / "rr.jsonl",
        [
            {
                "id": "x_0",
                "question": "q",
                "answer": "a",
                "supporting_facts": [],
                "type": "summary",
            },
            {"id": "x_1", "question": "", "answer": "a", "type": "single_hop"},
        ],
    )  # malformed
    res = D.load_ragrouter({"dom": rr}, {"seed": 1})
    assert [q.query_type for q in res.questions] == ["global"]
    assert len(res.rejected) == 1 and "reason" in res.rejected[0]  # reported, not dropped


def test_dedup_remaps_gold_ids_and_rejects_conflicts() -> None:
    docs = [
        Document(doc_id="a", text="same", source="s"),
        Document(doc_id="b", text="same", source="s"),
    ]
    qs = [Question(question_id="q", dataset="d", question="?", answers=["x"], gold_doc_ids=["b"])]
    kept, out_q, stats = deduplicate(docs, qs)
    assert [d.doc_id for d in kept] == ["a"] and out_q[0].gold_doc_ids == ["a"]
    assert stats["duplicate_texts"] == 1
    with pytest.raises(DataError):
        deduplicate(
            [
                Document(doc_id="a", text="x", source="s"),
                Document(doc_id="a", text="y", source="s"),
            ],
            [],
        )
    with pytest.raises(DataError):
        deduplicate([], qs + qs)


def test_split_is_stratified_deterministic_and_disjoint() -> None:
    qs = [
        Question(
            question_id=f"q{i}",
            dataset="d",
            question="?",
            answers=["a"],
            query_type=["single_hop", "multi_hop"][i % 2],
        )
        for i in range(100)
    ]
    s1, s2 = stratified_split(qs), stratified_split(list(reversed(qs)))
    assert s1 == s2 and len(s1["dev"]) == 20
    assert not set(s1["dev"]) & set(s1["test"])
    dev_types = [int(x[1:]) % 2 for x in s1["dev"]]
    assert dev_types.count(0) == dev_types.count(1) == 10


def test_reference_validation_and_store_checksums(tmp_path: Path) -> None:
    bad = [Question(question_id="q", dataset="d", question="?", answers=[], gold_doc_ids=["zzz"])]
    probs = validate_references([Document(doc_id="a", text="t", source="s")], bad)
    assert len(probs) == 2
    m = write_processed(
        "synthetic",
        synthetic_load_result(),
        tmp_path / "syn",
        params={"seed": 13},
        sources={},
        synthetic=True,
    )
    assert m["synthetic"] and m["counts"]["questions"] == 15
    store = DatasetStore(tmp_path / "syn")
    store.verify()
    assert len(store.questions("dev")) + len(store.questions("test")) == 15
    with pytest.raises(DataError):
        store.questions("train")
    (tmp_path / "syn" / "corpus.jsonl").write_text("tampered\n")
    with pytest.raises(DataError, match="checksum"):
        store.verify()
    with pytest.raises(DataError):
        DatasetStore(tmp_path / "nope")
    with pytest.raises(DataError):
        D.get_spec("unknown")


def test_offline_pipeline_never_sees_gold_fields() -> None:
    q = Question(
        question_id="q",
        dataset="d",
        question="Where?",
        answers=["secret"],
        gold_doc_ids=["g"],
        gold_evidence=["e"],
    )
    query = q.to_query()
    assert set(vars(query)) == {"query_id", "text"}


def test_paraphrased_evidence_disables_string_evidence_recall(tmp_path: Path) -> None:
    from elevate_rag.data.datasets import LoadResult
    from elevate_rag.data.prepare import evidence_verbatim_rate

    docs = [
        Document(
            doc_id="a", text="Basal cell carcinoma is the most common skin cancer.", source="s"
        )
    ]
    verbatim = [
        Question(
            question_id="v",
            dataset="d",
            question="?",
            answers=["x"],
            gold_evidence=["the most common skin cancer"],
        )
    ]
    para = [
        Question(
            question_id="p",
            dataset="d",
            question="?",
            answers=["x"],
            gold_evidence=["BCC is the most frequent cutaneous malignancy"],
        )
    ]
    assert evidence_verbatim_rate(docs, verbatim) == 1.0
    assert evidence_verbatim_rate(docs, para) == 0.0
    assert evidence_verbatim_rate(docs, []) is None
    write_processed("p", LoadResult(docs, para), tmp_path / "p", params={}, sources={})
    assert not DatasetStore(tmp_path / "p").string_evidence_valid
    write_processed("v", LoadResult(docs, verbatim), tmp_path / "v", params={}, sources={})
    assert DatasetStore(tmp_path / "v").string_evidence_valid


def test_download_reuses_verified_files_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from huggingface_hub import HfApi

    from elevate_rag.data.prepare import download

    spec = D.get_spec("multihop_rag")
    for rf in spec.files.values():
        dest = tmp_path / "hf" / rf.repo_id.replace("/", "__") / rf.revision / rf.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"[]")
        meta = {"size": 2, "sha256": hashlib.sha256(b"[]").hexdigest(), "blob_id": None}
        dest.with_name(dest.name + ".meta.json").write_text(json.dumps(meta))

    def offline(*a: object, **k: object) -> None:
        raise AssertionError("network must not be touched when files are verified")

    monkeypatch.setattr(HfApi, "get_paths_info", offline)
    paths = download(spec, tmp_path)
    assert set(paths) == {"corpus", "queries"} and all(p.exists() for p in paths.values())


def test_graphrag_bench_shared_ids_are_disambiguated_not_dropped(tmp_path: Path) -> None:
    (tmp_path / "c.json").write_text(json.dumps([{"corpus_name": "N", "context": "Text."}]))
    (tmp_path / "q.json").write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "question": "first?",
                    "answer": "a",
                    "question_type": "Complex Reasoning",
                },
                {
                    "id": "x",
                    "question": "second?",
                    "answer": "b",
                    "question_type": "Complex Reasoning",
                },
                {"id": "y", "question": "third?", "answer": "c", "question_type": "Fact Retrieval"},
            ]
        )
    )
    r = D.load_grb_novel(
        {"corpus": tmp_path / "c.json", "questions": tmp_path / "q.json"}, {"seed": 1}
    )
    ids = sorted(q.question_id for q in r.questions)
    assert ids == ["grb-x#1", "grb-x#2", "grb-y"]
    assert len(r.notes) == 2 and "shared by 2" in r.notes[0]


def test_ragrouter_accepts_both_evidence_shapes(tmp_path: Path) -> None:
    p = _jsonl(
        tmp_path / "q.jsonl",
        [
            {
                "id": "a",
                "question": "q1?",
                "answer": "x",
                "type": "single_hop",
                "supporting_facts": ["plain fact"],
            },
            {
                "id": "b",
                "question": "q2?",
                "answer": "y",
                "type": "multi_hop",
                "supporting_facts": [{"doc_id": 3, "text": " object fact "}, {"doc_id": 4}],
            },
        ],
    )
    r = D.load_ragrouter({"dom": p}, {"seed": 1})
    assert not r.rejected
    assert [q.gold_evidence for q in r.questions] == [["plain fact"], ["object fact"]]
