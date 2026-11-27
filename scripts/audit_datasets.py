"""Verify downloaded inputs, reproduce normalization, and audit split/evidence integrity."""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from elevate_rag.data import dataset_manifest, documents_at, file_hash, questions_at, write_json
from elevate_rag.datasets import DATASETS, PUBLIC_FILES, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase2/dataset-audit.json"))
    parser.add_argument("--refresh", choices=DATASETS, action="append", default=[])
    args = parser.parse_args()
    audit = {}
    split_texts: dict[str, dict[str, set[str]]] = {
        "dev": defaultdict(set),
        "heldout": defaultdict(set),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        if name not in PUBLIC_FILES:
            continue
        output = args.data / name
        original = dataset_manifest(output)
        raw_root = args.data / "input" / ("multihop" if name == "multihop-rag" else name)
        paths = [raw_root / entry for entry in PUBLIC_FILES[name]]
        indexed = paths if name != "ragrouter-bench" else [raw_root]
        raw = {f"{number}:{path.name}": file_hash(path) for number, path in enumerate(paths)}
        if raw != original["inputs"]:
            raise ValueError(f"{name}: downloaded file checksums differ from the manifest")
        if name in args.refresh:
            backup = args.out.parent / "prior-data" / name
            backup.mkdir(parents=True, exist_ok=True)
            for filename in ("manifest.json", "dev.jsonl", "heldout.jsonl"):
                target = backup / filename
                if not target.exists():
                    target.write_bytes((output / filename).read_bytes())
            prepare_dataset(name, indexed, output, original["seed"], original["version"])
        manifest = dataset_manifest(output)
        with tempfile.TemporaryDirectory(dir=args.out.parent, prefix="rebuild-") as scratch:
            repeated = prepare_dataset(
                name, indexed, Path(scratch), manifest["seed"], manifest["version"]
            )
        if repeated != manifest:
            raise ValueError(f"{name}: regenerated manifest differs")
        docs = {doc.doc_id: len(doc.text) for doc in documents_at(output)}
        dev, held = (questions_at(output, split) for split in ("dev", "heldout"))
        questions = dev + held
        groups = [
            Counter(" ".join(q.text.casefold().split()) for q in split) for split in (dev, held)
        ]
        for split, group in zip(("dev", "heldout"), groups, strict=True):
            for question_text in group:
                split_texts[split][question_text].add(name)
        overlap = groups[0].keys() & groups[1].keys()
        if (
            overlap
            or len(docs) != manifest["documents"]
            or len(questions) != len({q.question_id for q in questions})
        ):
            raise ValueError(f"{name}: overlap or inconsistent identifier counts")
        if any(
            e.doc_id not in docs or (e.end is not None and e.end > docs[e.doc_id])
            for q in questions
            for e in q.evidence
        ):
            raise ValueError(f"{name}: invalid evidence mapping")
        audit[name] = {
            "version": manifest["version"],
            "sources": PUBLIC_FILES[name],
            "inputs": raw,
            "normalized_files": manifest["files"],
            "documents": len(docs),
            "splits": {"dev": len(dev), "heldout": len(held)},
            "duplicate_question_text_rows": sum(
                count - 1 for count in (groups[0] + groups[1]).values()
            ),
            "cross_split_duplicate_text_groups": len(overlap),
            "duplicate_document_text_rows": manifest["duplicate_text_rows"],
            "answerability": dict(Counter(str(q.answerable) for q in questions)),
            "query_types": dict(Counter(q.query_type for q in questions)),
            "questions_with_mapped_evidence": sum(bool(q.evidence) for q in questions),
            "questions_with_evidence_text": sum(bool(q.evidence_text) for q in questions),
            "questions_with_evidence_triples": sum(bool(q.evidence_triples) for q in questions),
            "byte_identical_rebuild": True,
        }
        print(name, json.dumps(audit[name]["splits"]), flush=True)
    write_json(args.out, audit)
    common_texts = split_texts["dev"].keys() & split_texts["heldout"].keys()
    pairs = Counter()
    main_only = 0
    for text in common_texts:
        dev_names, heldout_names = (split_texts[split][text] for split in ("dev", "heldout"))
        pairs.update(f"{a} dev → {b} heldout" for a in dev_names for b in heldout_names)
        if dev_names - {"ragrouter-bench"} and heldout_names - {"ragrouter-bench"}:
            main_only += 1
    write_json(
        args.out.with_name("cross-dataset-overlap.json"),
        {
            "normalization": "casefold plus whitespace collapse",
            "all_benchmarks": len(common_texts),
            "main_benchmarks_only": main_only,
            "pair_counts": dict(pairs),
        },
    )


if __name__ == "__main__":
    main()
