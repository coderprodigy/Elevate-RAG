"""HotpotQA, distractor setting (dev split, all level "hard").

Each question ships with 10 paragraphs (2 gold + 8 distractors). We pool the paragraphs of all
sampled questions into one corpus; otherwise retrieval would be trivial.
"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq

from hgrag.data.loaders.base import BuildInputs, BuiltDataset, require_file, seeded_sample
from hgrag.data.schema import Document, Query, QueryType
from hgrag.data.text import clean_text, doc_id_for

RAW_FILE = "distractor/validation-00000-of-00001.parquet"
_TYPE_MAP: dict[str, QueryType] = {"bridge": "multi_hop", "comparison": "comparison"}


def build_hotpotqa(inputs: BuildInputs) -> BuiltDataset:
    rows: list[dict[str, Any]] = pq.read_table(require_file(inputs.raw_dir / RAW_FILE)).to_pylist()
    sample = seeded_sample(rows, inputs.settings.sample_size, inputs.seed)

    documents: list[Document] = []
    queries: list[Query] = []
    for row in sample:
        context = row["context"]
        gold_titles = set(row["supporting_facts"]["title"])
        gold_ids: set[str] = set()
        for raw_title, sentences in zip(context["title"], context["sentences"], strict=True):
            title, text = clean_text(raw_title), clean_text("".join(sentences))
            doc = Document(
                doc_id=doc_id_for(title, text), title=title, text=text, source=inputs.name
            )
            documents.append(doc)
            if raw_title in gold_titles:
                gold_ids.add(doc.doc_id)
        queries.append(
            Query(
                qid=f"hotpotqa-{row['id']}",
                question=clean_text(row["question"]),
                answers=[row["answer"]],
                gold_doc_ids=sorted(gold_ids),
                query_type=_TYPE_MAP[row["type"]],
                native_type=row["type"],
                hop_count=2,
                meta={"level": row["level"]},
            )
        )
    notes = [
        f"Sampled {len(sample)} of {len(rows)} dev questions; paragraphs pooled and deduplicated."
    ]
    return BuiltDataset(documents, queries, notes)
