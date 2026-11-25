"""MuSiQue (``musique_full_v1.0_dev.jsonl``): answerable and unanswerable 2-4 hop questions.

In the "full" file every unanswerable question is a twin of an answerable one with a supporting
paragraph removed. Pooling paragraphs into one corpus could put that missing paragraph back
(through the twin, or another question that shares it), making the question answerable. So an
unanswerable question is only kept when:

* its answerable twin is not sampled, and
* none of its missing supporting paragraphs are anywhere in the pooled corpus.
"""

from __future__ import annotations

import random
from typing import Any

from hgrag.data.loaders.base import BuildInputs, BuiltDataset, read_json_lines, seeded_sample
from hgrag.data.schema import Document, Query
from hgrag.data.text import clean_text, doc_id_for

RAW_FILE = "musique_full_v1.0_dev.jsonl"
Record = dict[str, Any]


def build_musique(inputs: BuildInputs) -> BuiltDataset:
    records: list[Record] = read_json_lines(inputs.raw_dir / RAW_FILE)
    answerable = [r for r in records if r["answerable"]]
    twins = {r["id"]: r for r in answerable}

    chosen = seeded_sample(answerable, inputs.settings.sample_size, inputs.seed)
    chosen_ids = {r["id"] for r in chosen}
    pool = {p["paragraph_text"] for r in chosen for p in r["paragraphs"]}

    unanswerable, rejected = _pick_unanswerable(
        candidates=[r for r in records if not r["answerable"] and r["id"] not in chosen_ids],
        twins=twins,
        pool=pool,
        wanted=inputs.settings.unanswerable_sample_size or 0,
        seed=inputs.seed,
    )

    documents: list[Document] = []
    queries: list[Query] = []
    for record in chosen + unanswerable:
        docs = [_paragraph(p, inputs.name) for p in record["paragraphs"]]
        documents += docs
        queries.append(_query(record, docs))

    notes = [
        f"{len(chosen)} answerable of {len(answerable)}; {len(unanswerable)} unanswerable "
        f"({rejected} candidates rejected because their missing evidence was in the pooled corpus "
        "or their answerable twin was sampled)."
    ]
    return BuiltDataset(documents, queries, notes)


def _pick_unanswerable(
    candidates: list[Record], twins: dict[str, Record], pool: set[str], wanted: int, seed: int
) -> tuple[list[Record], int]:
    """Greedy, seeded selection that keeps every chosen question genuinely unanswerable."""
    pool = set(pool)  # local copy: grows as questions are picked
    picked: list[Record] = []
    forbidden: set[str] = set()  # missing paragraphs of picked questions: must stay out of pool
    rejected = 0
    shuffled = random.Random(seed).sample(candidates, len(candidates))  # unbiased hop mix
    for record in shuffled:
        if len(picked) == wanted:
            break
        own = {p["paragraph_text"] for p in record["paragraphs"]}
        twin_support = {
            p["paragraph_text"] for p in twins[record["id"]]["paragraphs"] if p["is_supporting"]
        }
        missing = twin_support - own
        if missing & pool or own & forbidden:
            rejected += 1
            continue
        picked.append(record)
        pool |= own
        forbidden |= missing
    return picked, rejected


def _paragraph(paragraph: Record, source: str) -> Document:
    title, text = clean_text(paragraph["title"]), clean_text(paragraph["paragraph_text"])
    return Document(doc_id=doc_id_for(title, text), title=title, text=text, source=source)


def _query(record: Record, docs: list[Document]) -> Query:
    hop_prefix = record["id"].split("__")[0]  # e.g. "3hop1"
    if not record["answerable"]:
        return Query(
            qid=f"musique-{record['id']}-unans",
            question=clean_text(record["question"]),
            answers=[],
            query_type="unanswerable",
            native_type=f"{hop_prefix}-unanswerable",
            hop_count=int(hop_prefix[0]),
            answerable=False,
        )
    gold = [
        doc.doc_id for doc, p in zip(docs, record["paragraphs"], strict=True) if p["is_supporting"]
    ]
    return Query(
        qid=f"musique-{record['id']}",
        question=clean_text(record["question"]),
        answers=[record["answer"], *record["answer_aliases"]],
        gold_doc_ids=sorted(set(gold)),
        query_type="multi_hop",
        native_type=hop_prefix,
        hop_count=int(hop_prefix[0]),
    )
