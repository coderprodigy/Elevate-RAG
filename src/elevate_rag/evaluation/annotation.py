"""Human-annotation tooling: judge validation (FR-E9), triple-audit agreement (FR-G9) and the
error analysis sample (FR-E11).

Nothing here fabricates labels. Export functions write CSVs with an empty `label` column for
humans; agreement functions refuse to run on missing labels.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from elevate_rag.graph.quality import AUDIT_LABELS, read_labels
from elevate_rag.stats import cohen_kappa

JUDGE_LABELS = ("correct", "partial", "wrong")
KAPPA_GATE = 0.6

FAILURE_CAUSES = (
    "retrieval_miss",
    "extraction_miss",
    "wrong_route",
    "context_overload",
    "generation_error",
    "judge_error",
)


def export_judge_sample(
    scored: Sequence[dict[str, Any]],
    questions: dict[str, Any],
    path: Path,
    n: int = 100,
    seed: int = 13,
) -> int:
    """Random sample for two annotators. The judge's verdict is *not* shown to annotators."""
    pool = sorted(
        (r for r in scored if r["scores"].get("judge") in JUDGE_LABELS),
        key=lambda r: (r.get("run_id", ""), r["question_id"], r.get("mode", "")),
    )
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "question", "reference", "answer", "label"])
        w.writeheader()
        for i, r in enumerate(sample):
            q = questions[r["question_id"]]
            w.writerow(
                {
                    "item_id": f"j{i:03d}|{_item_key(r)}",
                    "question": q.question,
                    "reference": " | ".join(q.answers),
                    "answer": r["answer"],
                    "label": "",
                }
            )
    return len(sample)


def _item_key(r: dict[str, Any]) -> str:
    """Run + question + mode: a sample may span runs that answer the same question in the same
    mode (e.g. a router routing to `local` and the `local` variant), so the run id is needed."""
    return f"{r.get('run_id', '')}|{r['question_id']}|{r.get('mode', '')}"


def judge_agreement(
    sample_csv: Path, annotator_a: Path, annotator_b: Path, scored: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    a = read_labels(annotator_a, "item_id", JUDGE_LABELS)
    b = read_labels(annotator_b, "item_id", JUDGE_LABELS)
    ids = sorted(set(a) & set(b))
    if len(ids) < len(a) or len(ids) < len(b):
        raise ValueError("annotator files cover different items")
    with sample_csv.open(encoding="utf-8") as fh:
        sample_ids = {row["item_id"] for row in csv.DictReader(fh)}
    if set(ids) != sample_ids:
        raise ValueError(
            f"annotations cover {len(ids)} items but the sample has {len(sample_ids)}; "
            "label every row of the exported sample (and only those)"
        )
    verdict = {_item_key(r): r["scores"]["judge"] for r in scored}
    judge = [verdict.get(i.split("|", 1)[1]) for i in ids]
    if any(j is None for j in judge):
        raise ValueError("some annotated items are not in the scored results")
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    # human consensus = both agree; disagreements excluded from judge-vs-human kappa
    agree = [i for i, (x, y) in enumerate(zip(la, lb, strict=True)) if x == y]
    k_jh = cohen_kappa([la[i] for i in agree], [str(judge[i]) for i in agree]) if agree else None
    return {
        "n": len(ids),
        "kappa_annotators": cohen_kappa(la, lb),
        "kappa_judge_vs_consensus": k_jh,
        "n_consensus": len(agree),
        "gate_passed": k_jh is not None and k_jh >= KAPPA_GATE,
        "gate": KAPPA_GATE,
    }


def triple_audit_agreement(annotator_a: Path, annotator_b: Path) -> dict[str, Any]:
    a = read_labels(annotator_a, "triple_id", AUDIT_LABELS)
    b = read_labels(annotator_b, "triple_id", AUDIT_LABELS)
    ids = sorted(set(a) & set(b))
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    return {
        "n": len(ids),
        "kappa": cohen_kappa(la, lb),
        "precision_strict_a": la.count("correct") / len(la),
        "precision_strict_b": lb.count("correct") / len(lb),
        "labels_a": dict(Counter(la)),
        "labels_b": dict(Counter(lb)),
    }


def suggest_failure_cause(
    row: dict[str, Any], gold_doc_ids: Sequence[str], oracle_mode: str | None = None
) -> str:
    """Heuristic *suggestion* to speed up manual tagging — humans confirm the final label."""
    retrieved_docs = {c.split("::c")[0] for c in row.get("retrieved_chunk_ids", [])}
    context_docs = {c.split("::c")[0] for c in row.get("context_item_ids", [])}
    if oracle_mode and row.get("route") and oracle_mode != row["mode"]:
        return "wrong_route"
    if gold_doc_ids and not set(gold_doc_ids) & retrieved_docs:
        kg = [it for it in row.get("context_items", []) if it["kind"] != "chunk"]
        return (
            "extraction_miss" if row["mode"] in ("local", "global") and not kg else "retrieval_miss"
        )
    if (
        gold_doc_ids
        and set(gold_doc_ids) <= retrieved_docs
        and not (set(gold_doc_ids) <= context_docs)
    ):
        return "context_overload"
    return "generation_error"


def export_failure_sample(
    scored: Sequence[dict[str, Any]],
    questions: dict[str, Any],
    path: Path,
    n: int = 50,
    seed: int = 13,
    oracle: dict[str, str | None] | None = None,
) -> int:
    fails = sorted(
        (r for r in scored if r["scores"].get("correct") is False), key=lambda r: r["question_id"]
    )
    sample = random.Random(seed).sample(fails, min(n, len(fails)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "item_id",
                "mode",
                "question",
                "reference",
                "answer",
                "suggested_cause",
                "label",
            ],
        )
        w.writeheader()
        for r in sample:
            q = questions[r["question_id"]]
            w.writerow(
                {
                    "item_id": r["question_id"],
                    "mode": r["mode"],
                    "question": q.question,
                    "reference": " | ".join(q.answers),
                    "answer": r["answer"],
                    "suggested_cause": suggest_failure_cause(
                        r, q.gold_doc_ids, (oracle or {}).get(r["question_id"])
                    ),
                    "label": "",
                }
            )
    return len(sample)


def failure_cause_counts(labelled_csv: Path) -> dict[str, int]:
    labels = read_labels(labelled_csv, "item_id", FAILURE_CAUSES)
    return dict(Counter(labels.values()))
