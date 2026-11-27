"""Download, normalise, validate, deduplicate, split and manifest datasets (FR-D2–FR-D4).

Layout written for dataset `<name>` under `<data_dir>/processed/<name>/`:
  corpus.jsonl      normalised Documents
  questions.jsonl   normalised Questions (offline-only fields included)
  splits.json       {"dev": [...ids], "test": [...ids]} — 20 % dev, stratified by query type
  manifest.json     sources + revisions + sha256, parameters, counts, rejected records, checksums
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from elevate_rag.data.datasets import DatasetSpec, LoadResult, get_spec
from elevate_rag.types import Document, Question
from elevate_rag.utils import DataError, file_sha256, read_jsonl, write_json, write_jsonl

log = logging.getLogger(__name__)
LOADER_VERSION = "1"


def _git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, size: int | None, sha256: str | None, blob_id: str | None) -> bool:
    if not path.exists() or (size is not None and path.stat().st_size != size):
        return False
    if sha256:
        return file_sha256(path) == sha256
    if blob_id:
        return _git_blob_sha1(path) == blob_id
    return True


def download(
    spec: DatasetSpec, raw_dir: Path, *, timeout_s: float = 60.0, max_attempts: int = 8
) -> dict[str, Path]:
    """Fetch each pinned file over HTTPS with resume, verified against the Hub's LFS SHA-256
    (large files) or git blob SHA-1 (small files). Verified files are reused; a file that fails
    verification is re-downloaded, never used."""
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    out = {}
    for key, rf in spec.files.items():
        dest = raw_dir / "hf" / rf.repo_id.replace("/", "__") / rf.revision / rf.filename
        meta_path = dest.with_name(dest.name + ".meta.json")
        # The revision is pinned, so its size/hash never change: cache them so re-preparing
        # works offline. The file itself is still re-hashed against them every time.
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            info = api.get_paths_info(
                rf.repo_id, [rf.filename], revision=rf.revision, repo_type="dataset"
            )[0]
            if not isinstance(info, RepoFile):
                raise DataError(f"{rf.repo_id}/{rf.filename} is not a file at {rf.revision}")
            meta = {
                "size": info.size,
                "sha256": info.lfs.sha256 if info.lfs else None,
                "blob_id": info.blob_id,
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta))
        size, sha256 = meta["size"], meta["sha256"]
        blob_id = None if sha256 else meta["blob_id"]
        if not _verify(dest, size, sha256, blob_id):
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            url = (
                f"https://huggingface.co/datasets/{rf.repo_id}/resolve/{rf.revision}/{rf.filename}"
            )
            headers = (
                {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
                if os.getenv("HF_TOKEN")
                else {}
            )
            for attempt in range(1, max_attempts + 1):
                have = part.stat().st_size if part.exists() else 0
                rng = {"Range": f"bytes={have}-"} if have else {}
                try:
                    with httpx.stream(
                        "GET",
                        url,
                        headers={**headers, **rng},
                        follow_redirects=True,
                        timeout=timeout_s,
                    ) as r:
                        if r.status_code == 416:  # already complete
                            break
                        r.raise_for_status()
                        mode = "ab" if have and r.status_code == 206 else "wb"
                        with part.open(mode) as fh:
                            for block in r.iter_bytes(1 << 20):
                                fh.write(block)
                    break
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    log.warning(
                        "%s: download attempt %d/%d failed: %s",
                        rf.filename,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt == max_attempts:
                        raise DataError(f"could not download {url}: {exc}") from exc
            part.replace(dest)
            if not _verify(dest, size, sha256, blob_id):
                dest.unlink()
                raise DataError(f"{rf.repo_id}/{rf.filename}: checksum mismatch after download")
        out[key] = dest
    return out


def deduplicate(
    docs: list[Document], questions: list[Question]
) -> tuple[list[Document], list[Question], dict[str, int]]:
    """Drop exact-duplicate texts (keep first by doc_id) and remap gold IDs; reject conflicting
    duplicate IDs."""
    by_id: dict[str, Document] = {}
    for d in docs:
        if d.doc_id in by_id and by_id[d.doc_id].text != d.text:
            raise DataError(f"duplicate doc_id {d.doc_id} with different text")
        by_id[d.doc_id] = d
    by_text: dict[str, str] = {}
    remap: dict[str, str] = {}
    kept = []
    for d in sorted(by_id.values(), key=lambda d: d.doc_id):
        h = hashlib.sha256(d.text.strip().encode()).hexdigest()
        if h in by_text:
            remap[d.doc_id] = by_text[h]
        else:
            by_text[h] = d.doc_id
            kept.append(d)
    qids: set[str] = set()
    out_q = []
    for q in questions:
        if q.question_id in qids:
            raise DataError(f"duplicate question_id {q.question_id}")
        qids.add(q.question_id)
        if remap:
            q = q.model_copy(
                update={"gold_doc_ids": sorted({remap.get(g, g) for g in q.gold_doc_ids})}
            )
        out_q.append(q)
    stats = {"duplicate_ids": len(docs) - len(by_id), "duplicate_texts": len(remap)}
    return kept, out_q, stats


def stratified_split(
    questions: list[Question], dev_fraction: float = 0.2, seed: int = 13
) -> dict[str, list[str]]:
    """Deterministic dev/test split, stratified by query type and answerability."""
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for q in questions:
        groups[(q.query_type, q.answerable)].append(q.question_id)
    dev: list[str] = []
    test: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key], key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        n_dev = round(len(ids) * dev_fraction)
        dev += ids[:n_dev]
        test += ids[n_dev:]
    return {"dev": sorted(dev), "test": sorted(test)}


def validate_references(docs: list[Document], questions: list[Question]) -> list[str]:
    """Problems that would silently corrupt retrieval metrics."""
    ids = {d.doc_id for d in docs}
    problems = []
    for q in questions:
        missing = [g for g in q.gold_doc_ids if g not in ids]
        if missing and docs:
            problems.append(f"{q.question_id}: gold docs not in corpus: {missing[:3]}")
        if q.answerable and not q.answers:
            problems.append(f"{q.question_id}: answerable question without reference answer")
    return problems


EVIDENCE_VERBATIM_MIN = 0.9


def evidence_verbatim_rate(
    docs: list[Document], questions: list[Question], sample: int = 2000
) -> float | None:
    """Share of gold evidence strings that occur (normalised) in the corpus. String-match
    evidence recall is only valid when this is high; paraphrased evidence (e.g. GraphRAG-Bench)
    needs an LLM-judged evidence metric instead."""
    from elevate_rag.evaluation.metrics import normalize_answer

    ev = [e for q in questions for e in q.gold_evidence][:sample]
    if not ev:
        return None
    corpus = "\n".join(normalize_answer(d.text) for d in docs)
    return sum(normalize_answer(e) in corpus for e in ev) / len(ev)


def write_processed(
    name: str,
    result: LoadResult,
    out_dir: Path,
    *,
    params: dict[str, Any],
    sources: dict[str, Any],
    dev_fraction: float = 0.2,
    license: str = "",
    description: str = "",
    synthetic: bool = False,
) -> dict[str, Any]:
    docs, questions, dedup = deduplicate(result.documents, result.questions)
    if not questions:
        raise DataError(f"{name}: no valid questions")
    problems = validate_references(docs, questions)
    if any("not in corpus" in p for p in problems):
        raise DataError(f"{name}: {len(problems)} reference problems, e.g. {problems[:3]}")
    splits = stratified_split(questions, dev_fraction, params.get("seed", 13))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "corpus.jsonl", docs)
    write_jsonl(out_dir / "questions.jsonl", questions)
    write_json(out_dir / "splits.json", splits)
    manifest = {
        "dataset": name,
        "synthetic": synthetic,
        "description": description,
        "license": license,
        "loader_version": LOADER_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sources": sources,
        "params": params,
        "dev_fraction": dev_fraction,
        "counts": {
            "documents": len(docs),
            "questions": len(questions),
            "dev": len(splits["dev"]),
            "test": len(splits["test"]),
            "rejected_records": len(result.rejected),
            **dedup,
        },
        "query_types": dict(Counter(q.query_type for q in questions)),
        "evidence_verbatim_rate": evidence_verbatim_rate(docs, questions),
        "answerable": dict(Counter(str(q.answerable) for q in questions)),
        "warnings": problems[:50],
        "notes": result.notes,
        "rejected_examples": result.rejected[:20],
        "checksums": {
            f: file_sha256(out_dir / f) for f in ("corpus.jsonl", "questions.jsonl", "splits.json")
        },
    }
    manifest["version"] = hashlib.sha256(
        json.dumps(manifest["checksums"], sort_keys=True).encode()
    ).hexdigest()[:12]
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def prepare_dataset(
    name: str,
    data_dir: Path,
    overrides: dict[str, Any] | None = None,
    out_name: str | None = None,
) -> dict[str, Any]:
    """`out_name` writes a differently-parameterised variant (e.g. a 20-question smoke subset)
    next to the full dataset instead of replacing it."""
    spec = get_spec(name)
    params = {**spec.defaults, **(overrides or {})}
    files = download(spec, data_dir / "raw")
    sources = {
        k: {
            "repo": spec.files[k].repo_id,
            "file": spec.files[k].filename,
            "revision": spec.files[k].revision,
            "sha256": file_sha256(p),
        }
        for k, p in files.items()
    }
    result = spec.loader(files, params)
    if result.rejected:
        log.warning("%s: %d malformed records rejected (see manifest)", name, len(result.rejected))
    target = out_name or name
    return write_processed(
        target,
        result,
        data_dir / "processed" / target,
        params={**params, "base_dataset": name} if out_name else params,
        sources=sources,
        license=spec.license,
        description=spec.description + (f" — variant {out_name!r}" if out_name else ""),
    )


# --------------------------------------------------------------------------- reading back


class DatasetStore:
    """Read access to a prepared dataset; verifies checksums against the manifest."""

    def __init__(self, path: Path) -> None:
        self.path = path
        mf = path / "manifest.json"
        if not mf.exists():
            raise DataError(f"dataset not prepared: {path} (run `elevate-rag data prepare`)")
        self.manifest: dict[str, Any] = json.loads(mf.read_text())

    @classmethod
    def open(cls, data_dir: Path | str, name: str) -> DatasetStore:
        return cls(Path(data_dir) / "processed" / name)

    @property
    def string_evidence_valid(self) -> bool:
        """False when gold evidence is paraphrased → string evidence recall is undefined."""
        rate = self.manifest.get("evidence_verbatim_rate")
        return rate is not None and rate >= EVIDENCE_VERBATIM_MIN

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    def verify(self) -> None:
        for f, expected in self.manifest["checksums"].items():
            if file_sha256(self.path / f) != expected:
                raise DataError(f"{self.path / f}: checksum mismatch — re-run prepare")

    def documents(self) -> list[Document]:
        return [Document(**r) for r in read_jsonl(self.path / "corpus.jsonl")]

    def questions(self, split: str | None = None) -> list[Question]:
        qs = [Question(**r) for r in read_jsonl(self.path / "questions.jsonl")]
        if split is None or split == "all":
            return qs
        splits = json.loads((self.path / "splits.json").read_text())
        if split not in splits:
            raise DataError(f"unknown split {split!r}")
        keep = set(splits[split])
        return [q for q in qs if q.question_id in keep]
