"""Natural Questions, from DPR's ``biencoder-nq-dev.json.gz``.

Corpus = gold (positive) passages of the sampled questions plus up to
``max_distractors_per_question`` DPR hard negatives each.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

from hgrag.data.loaders.base import BuildInputs, BuiltDataset, require_file, seeded_sample
from hgrag.data.schema import Document, Query
from hgrag.data.text import clean_text, doc_id_for, qid_for

RAW_FILE = "biencoder-nq-dev.json.gz"


def build_nq(inputs: BuildInputs) -> BuiltDataset:
    with gzip.open(require_file(inputs.raw_dir / RAW_FILE), "rt", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = json.load(handle)

    usable = [r for r in records if r["positive_ctxs"]]
    sample = seeded_sample(usable, inputs.settings.sample_size, inputs.seed)
    max_distractors = inputs.settings.max_distractors_per_question

    documents: list[Document] = []
    queries: list[Query] = []
    for record in sample:
        gold = [_passage(ctx, inputs.name) for ctx in record["positive_ctxs"]]
        distractors = [
            _passage(ctx, inputs.name) for ctx in record["hard_negative_ctxs"][:max_distractors]
        ]
        documents += gold + distractors
        question = clean_text(record["question"])
        queries.append(
            Query(
                qid=qid_for("nq", question),
                question=question,
                answers=list(record["answers"]),
                gold_doc_ids=sorted({d.doc_id for d in gold}),
                query_type="single_hop",
                native_type="nq",
                hop_count=1,
            )
        )
    notes = [
        f"Sampled {len(sample)} of {len(usable)} dev questions; "
        f"up to {max_distractors} hard negatives each as distractors."
    ]
    return BuiltDataset(documents, queries, notes)


def _passage(ctx: dict[str, Any], source: str) -> Document:
    title, text = clean_text(ctx["title"]), clean_text(ctx["text"])
    return Document(
        doc_id=doc_id_for(title, text),
        title=title,
        text=text,
        source=source,
        meta={"wiki_passage_id": ctx.get("passage_id")},
    )
