"""Shared corpus and evaluation-only annotation contracts."""

import fcntl
import hashlib
import html
import json
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


def stable_hash(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def artifact_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"artifact is already in use: {path.parent}") from exc
        yield


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            f.write(content)
            f.flush()
            f.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            for value in values:
                f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            f.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                yield row
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str

    def __post_init__(self) -> None:
        for field in (self.doc_id, self.title, self.text, self.source):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("document ID, title, text and source must be nonempty strings")


@dataclass(frozen=True, slots=True)
class Evidence:
    doc_id: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if not self.doc_id or (self.start is None) != (self.end is None):
            raise ValueError("evidence needs a document ID and either zero or two offsets")
        if self.start is not None and self.end is not None and not 0 <= self.start < self.end:
            raise ValueError("invalid evidence offsets")


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    text: str
    answers: tuple[str, ...] = ()
    query_type: str = "unknown"
    answerable: bool | None = True
    evidence: tuple[Evidence, ...] = ()
    domain: str = ""
    evidence_text: tuple[str, ...] = ()
    evidence_triples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in (self.question_id, self.text, self.query_type):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("question ID, text and type must be nonempty strings")
        if self.answerable is not None and type(self.answerable) is not bool:
            raise ValueError(f"{self.question_id}: answerability must be boolean or null")
        if any(not isinstance(answer, str) or not answer.strip() for answer in self.answers):
            raise ValueError(f"{self.question_id}: reference answers must be nonempty strings")
        if self.answerable is False and self.answers:
            raise ValueError(f"{self.question_id}: unanswerable question has reference answers")

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Question":
        fields = dict(row)
        fields["answers"] = tuple(row.get("answers", ()))
        fields["evidence_text"] = tuple(row.get("evidence_text", ()))
        fields["evidence_triples"] = tuple(row.get("evidence_triples", ()))
        fields["evidence"] = tuple(Evidence(**item) for item in row.get("evidence", ()))
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    start: int
    end: int


class Tokenizer(Protocol):
    identity: str

    def count(self, text: str) -> int: ...

    def count_chat(self, system: str, user: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class CharacterTokenizer:
    """Network-free tokenizer for synthetic fixtures."""

    identity = "unicode-codepoint-v1"

    def count(self, text: str) -> int:
        return len(text)

    def count_chat(self, system: str, user: str) -> int:
        return self.count(system) + self.count(user) + 32

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


class HuggingFaceTokenizer:
    def __init__(self, model: str, revision: str, *, local_files_only: bool):
        from transformers import AutoTokenizer

        self.identity = f"{model}@{revision}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=local_files_only
        )
        if not self.tokenizer.is_fast:
            raise ValueError("chunking requires a fast tokenizer with character offsets")

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def count_chat(self, system: str, user: str) -> int:
        return len(
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

    def offsets(self, text: str) -> list[tuple[int, int]]:
        tokenized = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [(start, end) for start, end in tokenized["offset_mapping"] if start < end]


def split_questions(
    questions: Iterable[Question], seed: int
) -> tuple[list[Question], list[Question]]:
    """Allocate exactly one fifth to dev, approximating each joint stratum's share."""
    groups: dict[tuple[str, str, str], list[Question]] = defaultdict(list)
    for question in questions:
        groups[question.domain, question.query_type, str(question.answerable)].append(question)
    keys = sorted(groups, key=lambda key: (-(len(groups[key]) % 5), stable_hash(seed, key)))
    extra = sum(map(len, groups.values())) // 5 - sum(len(rows) // 5 for rows in groups.values())
    dev, heldout = [], []
    for position, key in enumerate(keys):
        rows = sorted(groups[key], key=lambda q: stable_hash(seed, q.question_id))
        size = len(rows) // 5 + int(position < extra)
        dev.extend(rows[:size])
        heldout.extend(rows[size:])
    # A benchmark can publish the same question with different document IDs.
    # Keep those rows together, preserving the original stratification where possible.
    texts = {
        q.question_id: " ".join(q.text.casefold().split()) for rows in groups.values() for q in rows
    }
    overlap = {texts[q.question_id] for q in dev} & {texts[q.question_id] for q in heldout}
    if overlap:
        target = len(dev)
        desired = Counter((q.domain, q.query_type, q.answerable) for q in dev)
        displaced = [q for q in dev if texts[q.question_id] in overlap]
        dev = [q for q in dev if texts[q.question_id] not in overlap]
        heldout.extend(displaced)
        available: dict[str, list[Question]] = defaultdict(list)
        for question in heldout:
            available[texts[question.question_id]].append(question)
        current = Counter((q.domain, q.query_type, q.answerable) for q in dev)
        ranked = sorted(
            available.values(),
            key=lambda rows: (
                len(rows) != 1,
                -sum(
                    desired[key] > current[key]
                    for key in ((q.domain, q.query_type, q.answerable) for q in rows)
                ),
                stable_hash(seed, texts[rows[0].question_id]),
            ),
        )
        moved: set[str] = set()
        for rows in ranked:
            if len(rows) <= target - len(dev):
                dev.extend(rows)
                moved.update(q.question_id for q in rows)
            if len(dev) == target:
                break
        heldout = [q for q in heldout if q.question_id not in moved]
    return sorted(dev, key=lambda q: q.question_id), sorted(heldout, key=lambda q: q.question_id)


def prepare(
    path: Path,
    documents: Iterable[Document],
    questions: Iterable[Question],
    *,
    dataset: str,
    version: str,
    seed: int,
    sources: dict[str, str],
    synthetic: bool = False,
) -> dict[str, Any]:
    docs: dict[str, Document] = {}
    duplicates = 0
    for document in documents:
        if document.doc_id in docs:
            if document != docs[document.doc_id]:
                raise ValueError(f"inconsistent document ID: {document.doc_id}")
            duplicates += 1
        docs[document.doc_id] = document
    if not docs:
        raise ValueError("empty corpus")
    question_list = list(questions)
    if not question_list:
        raise ValueError("empty question set")
    counts = Counter(question.question_id for question in question_list)
    if repeated := sorted(key for key in counts if counts[key] > 1):
        raise ValueError(f"duplicate question IDs: {repeated[:5]}")
    for question in question_list:
        for evidence in question.evidence:
            source_document = docs.get(evidence.doc_id)
            if source_document is None or (
                evidence.end is not None and evidence.end > len(source_document.text)
            ):
                raise ValueError(
                    f"{question.question_id}: unresolved supporting evidence {evidence}"
                )
    dev, heldout = split_questions(question_list, seed)
    write_jsonl(path / "documents.jsonl", (asdict(docs[key]) for key in sorted(docs)))
    write_jsonl(path / "dev.jsonl", map(asdict, dev))
    write_jsonl(path / "heldout.jsonl", map(asdict, heldout))
    manifest = {
        "schema": 1,
        "dataset": dataset,
        "version": version,
        "seed": seed,
        "synthetic": synthetic,
        "inputs": sources,
        "documents": len(docs),
        "duplicate_document_rows": duplicates,
        "duplicate_text_rows": len(docs) - len({doc.text for doc in docs.values()}),
        "questions": {"dev": len(dev), "heldout": len(heldout)},
        "missing_evidence_questions": sum(not question.evidence for question in question_list),
        "files": {
            name: file_hash(path / name)
            for name in ("documents.jsonl", "dev.jsonl", "heldout.jsonl")
        },
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def dataset_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported dataset manifest schema")
    for name in ("documents.jsonl", "dev.jsonl", "heldout.jsonl"):
        if file_hash(path / name) != manifest["files"][name]:
            raise ValueError(f"dataset integrity check failed for {name}")
    return manifest


def documents_at(path: Path) -> Iterator[Document]:
    for row in read_jsonl(path / "documents.jsonl"):
        yield Document(**row)


def questions_at(path: Path, split: str) -> list[Question]:
    if split not in {"dev", "heldout"}:
        raise ValueError(f"unsupported split: {split}")
    return [Question.from_row(row) for row in read_jsonl(path / f"{split}.jsonl")]


DISALLOWED_PASSAGE_CHARACTERS = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def canonical_passage(text: str) -> str:
    # Upstream graph ingestion decodes HTML escapes. Keep both paths on the same final text.
    while True:
        sanitized = DISALLOWED_PASSAGE_CHARACTERS.sub("", html.unescape(text)).strip()
        if sanitized == text:
            return text
        text = sanitized


def chunk_document(
    document: Document, tokenizer: Tokenizer, size: int, overlap: int, paragraph: bool = False
) -> Iterator[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("chunk size must be greater than the nonnegative overlap")
    regions = (
        list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", document.text, re.S))
        if paragraph
        else []
    )
    spans = [(match.start(), match.end()) for match in regions] or [(0, len(document.text))]
    for region_start, region_end in spans:
        passage = document.text[region_start:region_end]
        offsets = tokenizer.offsets(passage)
        sentence_ends = {match.start() for match in re.finditer(r"(?<=[.!?])\s+|\n{2,}", passage)}
        index = 0
        while index < len(offsets):
            end_index = min(index + size, len(offsets))
            if end_index < len(offsets):
                lower = index + max(overlap + 1, size // 2)
                for candidate in range(end_index, lower - 1, -1):
                    if offsets[candidate - 1][1] in sentence_ends:
                        end_index = candidate
                        break
            start = region_start + offsets[index][0]
            end = region_start + offsets[end_index - 1][1]
            text = canonical_passage(document.text[start:end])
            while tokenizer.count(text) > size and end_index > index + 1:
                end_index -= 1
                end = region_start + offsets[end_index - 1][1]
                text = canonical_passage(document.text[start:end])
            if tokenizer.count(text) > size:
                raise ValueError("tokenizer cannot represent a single chunk unit within budget")
            if text:
                start += len(document.text[start:end]) - len(document.text[start:end].lstrip())
                end = region_start + offsets[end_index - 1][1]
                end -= len(document.text[start:end]) - len(document.text[start:end].rstrip())
                yield Chunk(
                    "c-" + stable_hash(document.doc_id, start, end, text),
                    document.doc_id,
                    document.title,
                    text,
                    document.source,
                    start,
                    end,
                )
            if end_index == len(offsets):
                break
            index = max(index + 1, end_index - overlap)
