from pathlib import Path

from hgrag.data.io import DatasetFiles, write_jsonl
from hgrag.data.loaders.graphrag_bench import build_graphrag_bench_novel
from hgrag.data.loaders.hotpotqa import build_hotpotqa
from hgrag.data.loaders.multihop_rag import build_multihop_rag
from hgrag.data.loaders.musique import _pick_unanswerable, build_musique
from hgrag.data.loaders.nq import build_nq
from hgrag.data.loaders.ragrouter_bench import DOMAINS, build_ragrouter_bench
from hgrag.data.schema import Query

from .raw_fixtures import (
    make_inputs,
    musique_record,
    write_graphrag_bench,
    write_hotpotqa,
    write_json_lines,
    write_multihop_rag,
    write_nq,
)


def test_nq_samples_questions_and_caps_distractors(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "nq", sample_size=2, max_distractors_per_question=1)
    write_nq(inputs.raw_dir)
    built = build_nq(inputs)

    assert len(built.queries) == 2
    assert len(built.documents) == 4  # per question: 1 gold + 1 distractor
    doc_ids = {d.doc_id for d in built.documents}
    for query in built.queries:
        assert query.query_type == "single_hop" and query.hop_count == 1
        assert len(query.gold_doc_ids) == 1 and set(query.gold_doc_ids) <= doc_ids


def test_nq_sampling_is_deterministic(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "nq", sample_size=2)
    write_nq(inputs.raw_dir)
    assert [q.qid for q in build_nq(inputs).queries] == [q.qid for q in build_nq(inputs).queries]


def test_hotpotqa_pools_paragraphs_and_marks_gold(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "hotpotqa")
    write_hotpotqa(inputs.raw_dir)
    built = build_hotpotqa(inputs)

    by_title = {d.title: d for d in built.documents}
    assert by_title["A"].text == "A was built in 1900. It is old."
    comparison, bridge = built.queries
    assert comparison.query_type == "comparison" and bridge.query_type == "multi_hop"
    assert set(comparison.gold_doc_ids) == {by_title["A"].doc_id, by_title["B"].doc_id}
    # paragraph "A" and "C" appear under both questions → same content-hash ID
    assert len({d.doc_id for d in built.documents}) == 4


def test_musique_unanswerable_selection_keeps_questions_unanswerable() -> None:
    # Twins: "u1" lacks P1, "u2" lacks P4; "u3" contains P4, so u2 and u3 cannot both be kept.
    answerable = {
        "u1": musique_record("u1", True, [("P1", True), ("P2", True)]),
        "u2": musique_record("u2", True, [("P4", True), ("P5", True)]),
        "u3": musique_record("u3", True, [("P6", True), ("P7", True)]),
    }
    candidates = [
        musique_record("u1", False, [("P2", False)]),
        musique_record("u2", False, [("P5", False)]),
        musique_record("u3", False, [("P4", False), ("P7", False)]),  # lacks P6
    ]
    pool = {"P1"}  # P1 is already in the corpus → u1 must be rejected
    picked, rejected = _pick_unanswerable(candidates, answerable, pool, wanted=10, seed=750)

    assert pool == {"P1"}, "caller's pool must not be mutated"
    assert "u1" not in {r["id"] for r in picked}
    assert len(picked) == 1 and rejected == 2


def test_musique_never_keeps_twin_of_sampled_answerable(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "musique", unanswerable_sample_size=5)
    write_json_lines(
        inputs.raw_dir / "musique_full_v1.0_dev.jsonl",
        [
            musique_record("2hop__1_2", True, [("P1", True), ("P2", True), ("P3", False)]),
            musique_record("2hop__1_2", False, [("P2", False), ("P3", False)]),
            musique_record("3hop1__4_5_6", True, [("P4", True), ("P5", False)]),
        ],
    )
    built = build_musique(inputs)

    assert all(q.answerable for q in built.queries), "both twins' answerable halves were sampled"
    two_hop = next(q for q in built.queries if q.qid == "musique-2hop__1_2")
    assert two_hop.hop_count == 2 and two_hop.answers == ["ans", "alias"]
    assert len(two_hop.gold_doc_ids) == 2
    assert next(q for q in built.queries if q.native_type == "3hop1").hop_count == 3


def test_multihop_rag_maps_titles_and_null_queries(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "multihop_rag")
    write_multihop_rag(inputs.raw_dir)
    built = build_multihop_rag(inputs)

    assert "Second paragraph 0." in built.documents[0].text and "\n\n" in built.documents[0].text
    inference, null = built.queries
    assert inference.query_type == "multi_hop" and inference.hop_count == 2
    assert set(inference.gold_doc_ids) == {d.doc_id for d in built.documents}
    assert inference.gold_evidence == ["fact zero", "fact one"]
    assert null.query_type == "unanswerable" and not null.answerable and null.answers == []
    assert null.meta["original_answer"] == "Insufficient information."


def test_graphrag_bench_maps_levels_and_links_source(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "graphrag_bench_novel", raw_name="graphrag_bench")
    write_graphrag_bench(inputs.raw_dir)
    built = build_graphrag_bench_novel(inputs)

    types = [q.query_type for q in built.queries]
    assert types == ["single_hop", "multi_hop", "global", "creative", "single_hop"]
    novel_1 = next(d for d in built.documents if d.title == "Novel-1")
    assert built.queries[0].gold_doc_ids == [novel_1.doc_id]
    assert built.queries[-1].gold_doc_ids == []  # unknown source document
    assert "4/5 questions linked" in built.notes[0]


def test_ragrouter_drops_questions_we_evaluate_on(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, "ragrouter_bench")
    for domain in DOMAINS:
        write_json_lines(inputs.raw_dir / f"data/{domain}/Question.json", [])
    write_json_lines(
        inputs.raw_dir / "data/musique/Question.json",
        [
            {
                "id": "musique_0",
                "question": "Who is the spouse of X?",
                "answer": "Y",
                "supporting_facts": [],
                "type": "multi_hop",
            },
            {
                "id": "musique_1",
                "question": "Summarise the corpus.",
                "answer": "Z",
                "supporting_facts": ["a fact", {"doc_id": 3, "text": " a  passage "}],
                "type": "summary",
            },
        ],
    )
    evaluated = Query(
        qid="musique-1",
        question="who is the SPOUSE of x",
        answers=["Y"],
        query_type="multi_hop",
        native_type="2hop",
    )
    write_jsonl(DatasetFiles(inputs.processed_root / "musique").queries, [evaluated])

    built = build_ragrouter_bench(inputs)

    assert [q.qid for q in built.queries] == ["rr-musique_1"]
    assert built.queries[0].query_type == "global" and built.queries[0].meta["domain"] == "musique"
    assert built.queries[0].gold_evidence == ["a fact", "a passage"]  # both fact formats
    assert "Dropped 1" in built.notes[0] and "musique" in built.notes[0]
