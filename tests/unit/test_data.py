import json
from dataclasses import asdict

import pytest

from elevate_rag.data import (
    CharacterTokenizer,
    Document,
    Evidence,
    Question,
    chunk_document,
    dataset_manifest,
    prepare,
    questions_at,
    split_questions,
    write_jsonl,
)
from elevate_rag.datasets import graphrag, hotpot, multihop, musique, nq, ragrouter, rows


def test_chunk_overlap_and_original_offsets():
    doc = Document(
        "doc", "Title", "First sentence. Second statement! Last example? Another one.", "source"
    )
    chunks = list(chunk_document(doc, CharacterTokenizer(), 26, 6))
    assert len(chunks) > 2
    assert chunks[0].text == "First sentence."
    assert all(chunk.text == doc.text[chunk.start : chunk.end] for chunk in chunks)
    assert all(len(chunk.text) <= 26 for chunk in chunks)
    assert chunks == list(chunk_document(doc, CharacterTokenizer(), 26, 6))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    with pytest.raises(ValueError):
        list(chunk_document(doc, CharacterTokenizer(), 12, 12))


def test_paragraph_boundaries():
    doc = Document(
        "doc", "Title", "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", "source"
    )
    chunks = list(chunk_document(doc, CharacterTokenizer(), 128, 0, paragraph=True))
    assert [chunk.text for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_fixture_integrity_and_deterministic_partitions(data_path):
    manifest = dataset_manifest(data_path)
    assert manifest["questions"] == {"dev": 4, "heldout": 16}
    dev, heldout = questions_at(data_path, "dev"), questions_at(data_path, "heldout")
    assert split_questions([*reversed(dev), *reversed(heldout)], 42) == (dev, heldout)
    assert {q.question_id for q in dev}.isdisjoint(q.question_id for q in heldout)
    (data_path / "heldout.jsonl").write_text("")
    with pytest.raises(ValueError, match="integrity"):
        dataset_manifest(data_path)


def test_invalid_annotations(tmp_path):
    doc = Document("a", "Title", "paragraph", "source")
    with pytest.raises(ValueError, match="unresolved"):
        prepare(
            tmp_path,
            [doc],
            [Question("q", "Who?", evidence=(Evidence("missing"),))],
            dataset="test",
            version="1",
            seed=0,
            sources={},
        )
    with pytest.raises(ValueError, match="duplicate question"):
        prepare(
            tmp_path,
            [doc],
            [Question("q", "Who?")] * 2,
            dataset="test",
            version="1",
            seed=0,
            sources={},
        )
    with pytest.raises(ValueError, match="empty corpus"):
        prepare(tmp_path, [], [], dataset="test", version="1", seed=0, sources={})
    with pytest.raises(ValueError, match="unanswerable"):
        Question("q", "Who?", ("answer",), answerable=False)


def test_hotpot_support_offsets(tmp_path):
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "q",
                    "level": "hard",
                    "question": "Who lives there?",
                    "answer": "Ana",
                    "type": "bridge",
                    "context": [["Home", ["A house.", "Ana lives in it."]], ["Elsewhere", ["No."]]],
                    "supporting_facts": [["Home", 1]],
                }
            ]
        )
    )
    docs, [q] = hotpot(path, 42, 1)
    (evidence,) = q.evidence
    assert docs[0].doc_id == evidence.doc_id
    assert docs[0].text[evidence.start : evidence.end] == "Ana lives in it."
    assert q.query_type == "multi-hop"


def test_musique_both_labels_and_gold(tmp_path):
    path = tmp_path / "musique.jsonl"
    base = {
        "question": "Who?",
        "answer": "Ana",
        "answer_aliases": ["Ann"],
        "paragraphs": [
            {
                "title": "Evidence",
                "paragraph_text": "Ana lives there.",
                "is_supporting": True,
            }
        ],
    }
    write_jsonl(
        path,
        [
            {**base, "id": "2hop1", "answerable": True},
            {**base, "id": "2hop1", "answerable": False},
            {**base, "id": "2hop2", "answerable": True},
        ],
    )
    docs, questions = musique(path, 7, 1, 1)
    assert questions[0].answers == ("Ana", "Ann")
    assert questions[1].answers == ()
    assert [q.question_id for q in questions] == ["2hop2", "2hop1"]
    assert docs[0].doc_id == questions[0].evidence[0].doc_id


def test_ragrouter_reads_published_json_lines_and_document_ids(tmp_path):
    for domain in ("musique", "quality", "ultraDomain_legal", "graphragBench_medical"):
        root = tmp_path / domain
        write_jsonl(root / "Corpus.json", [{"id": 0, "title": "", "context": "Ana lives there."}])
        write_jsonl(
            root / "Question.json",
            [
                {
                    "id": "q",
                    "question": "Who lives there?",
                    "answer": "Ana",
                    "type": "single_hop",
                    "supporting_facts": [0, "Ana lives there."],
                }
            ],
        )
    docs, questions, paths = ragrouter(tmp_path)
    assert len(paths) == 8
    assert [doc.title for doc in docs] == [doc.source for doc in docs]
    assert len({doc.source for doc in docs}) == 4
    assert questions[-1].evidence == (Evidence(docs[-1].doc_id),)


def test_nq_support_and_distractors(tmp_path):
    path = tmp_path / "nq.jsonl"
    write_jsonl(
        path,
        [
            {
                "example_id": 123,
                "question_text": "Who?",
                "document_url": "https://example.test/test",
                "document_text": "<P> Ana lives there. </P> <P> Elsewhere. </P>",
                "long_answer_candidates": [
                    {"start_token": 0, "end_token": 5, "top_level": True},
                    {"start_token": 5, "end_token": 8, "top_level": True},
                ],
                "annotations": [
                    {
                        "long_answer": {"candidate_index": 0},
                        "short_answers": [{"start_token": 1, "end_token": 2}],
                        "yes_no_answer": "NONE",
                    }
                ],
            }
        ],
    )
    docs, [question] = nq(path, 0, 1)
    assert [doc.text for doc in docs] == ["Ana lives there.", "Elsewhere."]
    assert question.answers == ("Ana",)
    assert question.evidence == (Evidence(docs[0].doc_id),)


def test_multihop_evidence_and_graphrag_metadata(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    write_jsonl(corpus, [{"title": "Article", "body": "Ana lives there.", "url": "https://a.test"}])
    write_jsonl(
        queries,
        [
            {
                "query": "Who?",
                "answer": "Ana",
                "question_type": "temporal",
                "evidence_list": [
                    {"fact": "Ana lives there.", "url": "https://a.test"},
                ],
            }
        ],
    )
    docs, [question] = multihop(corpus, queries)
    assert question.evidence == (Evidence(docs[0].doc_id, 0, 16),)
    write_jsonl(corpus, [{"corpus_name": "Novel", "context": "Different text."}])
    write_jsonl(
        queries,
        [
            {
                "source": "Novel",
                "id": "q1",
                "question": "Who?",
                "answer": "Ana",
                "question_type": "Fact Retrieval",
                "evidence": ["Cannot be located"],
            }
        ],
    )
    _, [question] = graphrag(corpus, queries, "novel")
    assert question.evidence == ()
    assert question.evidence_text == ("Cannot be located",)
    assert question.question_id == "Novel:q1"


def test_malformed_jsonl_has_location(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(asdict(Document("a", "Title", "Text", "source"))) + "\n[]\n")
    with pytest.raises(ValueError, match=r"invalid.jsonl:2"):
        list(rows(path))
