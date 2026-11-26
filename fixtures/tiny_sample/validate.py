"""Validate the Tiny_Sample fixture. Exits non-zero with a message on the first failure.

Run from the repository root:  python fixtures/tiny_sample/validate.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
QUERY_TYPES = {"single-hop", "multi-hop", "comparison", "temporal", "global", "null"}
LABELS = {"correct", "partial", "wrong"}
REFUSAL = "insufficient information"
DELIM = "<|#|>"
COMPLETE = "<|COMPLETE|>"


def load(rel: str) -> list[dict[str, Any]]:
    lines = (HERE / rel).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"FAIL: {msg}")


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    corpus = load("corpus.jsonl")
    questions = load("questions.jsonl")
    extraction = load("fake_llm/extraction.jsonl")
    generation = load("fake_llm/generation.jsonl")
    judge = load("fake_llm/judge.jsonl")
    malformed = load("fake_llm/malformed.jsonl")

    # Normalised_Corpus schema and unique IDs
    doc_ids = [d["doc_id"] for d in corpus]
    check(len(doc_ids) == len(set(doc_ids)), "duplicate doc_id")
    for d in corpus:
        check(set(d) == {"doc_id", "title", "text", "source"}, f"corpus fields {d['doc_id']}")
        check(bool(d["text"].strip()), f"empty text {d['doc_id']}")
        # Each document must fit in one chunk even at the 256-token ablation size.
        check(len(d["text"].split()) <= 120, f"document too long {d['doc_id']}")
    docs = set(doc_ids)

    # Question_Set schema, coverage and referential integrity
    qfields = {
        "question_id",
        "question",
        "answers",
        "gold_passage_ids",
        "query_type",
        "answerable",
        "split",
    }
    check(0 < len(questions) <= 20, "question count must be 1..20")
    qids = [q["question_id"] for q in questions]
    check(len(qids) == len(set(qids)), "duplicate question_id")
    for q in questions:
        check(set(q) == qfields, f"question fields {q['question_id']}")
        check(q["query_type"] in QUERY_TYPES, f"bad query_type {q['question_id']}")
        check(q["split"] in (None, "dev", "test"), f"bad split {q['question_id']}")
        check(all(g in docs for g in q["gold_passage_ids"]), f"dangling gold {q['question_id']}")
        if q["query_type"] == "null":
            check(
                not q["answerable"] and not q["answers"] and not q["gold_passage_ids"],
                f"null question must be unanswerable with no gold {q['question_id']}",
            )
        else:
            check(
                q["answerable"] and q["answers"] and q["gold_passage_ids"],
                f"answerable question needs answers and gold {q['question_id']}",
            )
        if q["query_type"] in {"multi-hop", "comparison"}:
            check(len(q["gold_passage_ids"]) >= 2, f"needs >=2 gold docs {q['question_id']}")
    check({q["query_type"] for q in questions} == QUERY_TYPES, "not every Query_Type covered")
    check(manifest["num_questions"] == len(questions), "manifest question count")
    check(manifest["num_documents"] == len(corpus), "manifest document count")

    # Extraction: one per doc, raw LightRAG text consistent with structured fields
    check(sorted(e["doc_id"] for e in extraction) == sorted(docs), "extraction per doc")
    text_by_doc = {d["doc_id"]: d["text"] for d in corpus}
    entity_docs: dict[str, set[str]] = defaultdict(set)
    for e in extraction:
        check(e["match"] in text_by_doc[e["doc_id"]], f"match not in text {e['doc_id']}")
        lines = e["response"].split("\n")
        check(lines[-1] == COMPLETE, f"missing completion delimiter {e['doc_id']}")
        rows = [line.split(DELIM) for line in lines[:-1]]
        ents = [r for r in rows if r[0] == "entity"]
        rels = [r for r in rows if r[0] == "relation"]
        check(len(ents) + len(rels) == len(rows), f"unknown row type {e['doc_id']}")
        check(
            all(len(r) == 4 for r in ents) and all(len(r) == 5 for r in rels),
            f"bad tuple arity {e['doc_id']}",
        )
        check(
            len(ents) == len(e["entities"]) and len(rels) == len(e["relations"]),
            f"raw/structured mismatch {e['doc_id']}",
        )
        names = {x["name"] for x in e["entities"]}
        for r in e["relations"]:
            check(
                r["source"] in names and r["target"] in names,
                f"relation endpoint not an entity {e['doc_id']}: {r}",
            )
            check(0.0 < r["weight"] <= 1.0, f"weight out of range {e['doc_id']}")
        for n in names:
            entity_docs[n.casefold()].add(e["doc_id"])

    # Multi-hop gold docs must be linked through shared entities (graph path exists).
    doc_entities: dict[str, set[str]] = defaultdict(set)
    for name, ds in entity_docs.items():
        for d in ds:
            doc_entities[d].add(name)
    for q in questions:
        gold = q["gold_passage_ids"]
        if q["query_type"] == "multi-hop":
            for a, b in pairwise(gold):
                check(
                    bool(doc_entities[a] & doc_entities[b]),
                    f"multi-hop gold docs {a},{b} share no entity ({q['question_id']})",
                )

    # Generation and judge scripts
    gkeys = {(g["question_id"], g["variant"]) for g in generation}
    check(len(gkeys) == len(generation), "duplicate generation key")
    qmap = {q["question_id"]: q for q in questions}
    for g in generation:
        q = qmap.get(g["question_id"])
        check(q is not None, f"generation for unknown question {g['question_id']}")
        assert q is not None
        check(all(c in docs for c in g["cited_doc_ids"]), f"dangling citation {g}")
        if g["variant"] == "rag":
            check(
                set(g["cited_doc_ids"]) <= set(q["gold_passage_ids"]),
                f"rag cites non-gold doc {g['question_id']}",
            )
            if not q["answerable"]:
                check(g["answer"] == REFUSAL, f"null rag answer must refuse {g['question_id']}")
    check({(q["question_id"], "rag") for q in questions} <= gkeys, "missing rag responses")
    check({(j["question_id"], j["variant"]) for j in judge} == gkeys, "judge/generation keys")
    gans = {(g["question_id"], g["variant"]): g["answer"] for g in generation}
    for j in judge:
        check(j["prediction"] == gans[(j["question_id"], j["variant"])], f"prediction {j}")
        verdict = json.loads(j["response"])
        check(verdict["label"] in LABELS and bool(verdict["rationale"]), f"verdict {j}")
    check(
        {m["role"] for m in malformed} == {"extraction", "judge", "generation"},
        "malformed examples per role",
    )

    print(
        f"OK: {len(corpus)} docs, {len(questions)} questions, {len(extraction)} extractions, "
        f"{len(generation)} generations, {len(judge)} judgements, {len(malformed)} malformed"
    )


if __name__ == "__main__":
    main()
