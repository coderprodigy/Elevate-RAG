"""Audio/video clips → a normal Elevate-RAG dataset (docs/MEDIA_INGESTION_DESIGN.md).

Pure Python: discovery, licences, transcript filtering and caching, time windows, documents,
question mapping and the `ingest_media` orchestration. The heavy parts (PyAV probing and
faster-whisper transcription) live in `media_backends` and are passed in, so this module is
importable and testable without the `media` extra, and never loads PyTorch or FAISS.

Each clip becomes
* transcript windows (~`window_s` seconds, cut at segment boundaries, one segment of overlap),
  each small enough to stay one chunk, so a retrieved chunk maps to one time span;
* one clip card: its technical details as sentences.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from elevate_rag.data.datasets import LoadResult
from elevate_rag.data.prepare import write_processed
from elevate_rag.types import QUERY_TYPES, Document, Question
from elevate_rag.utils import DataError, file_sha256, stable_hash, write_json

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aiff", ".aif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm"})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
# Files that live next to the clips and are not media.
SIDECAR_FILES = frozenset(
    {"licences.csv", "questions.jsonl", "questions_v2.jsonl", "vocabulary.txt"}
)
# Whisper's own thresholds: a segment that is probably not speech *and* decoded with low
# confidence is dropped as a likely hallucination.
NO_SPEECH_MAX = 0.6
LOGPROB_MIN = -1.0
MEDIA_LOADER_VERSION = "media-v1"


# ------------------------------------------------------------------------------- data types


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class Transcript:
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]
    model: str
    revision: str | None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        return cls(**{**d, "segments": [Segment(**s) for s in d["segments"]]})


@dataclass
class ClipInfo:
    """Technical details of one clip (from `media_backends.probe`)."""

    kind: str  # audio | video
    duration_s: float
    container: str
    audio: dict[str, Any] | None = None  # codec, sample_rate, channels
    video: dict[str, Any] | None = None  # codec, width, height, fps
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Clip:
    path: Path
    file: str  # path relative to the source folder, POSIX style
    sha256: str
    bytes: int
    clip_id: str
    licence: dict[str, str]
    info: ClipInfo | None = None


class Transcriber(Protocol):
    model: str
    revision: str | None
    params: dict[str, Any]

    def transcribe(self, path: Path) -> Transcript: ...


ProbeFn = Callable[[Path], ClipInfo]
# (path, speech share 0-1 or None) -> music details (media_music.analyse)
MusicFn = Callable[[Path, float | None], dict[str, Any]]
# path -> (scenes with .start_s/.end_s/.keyframe_s/.caption, settings) (media_vision.scene_fn)
ScenesFn = Callable[[Path], tuple[list[Any], dict[str, Any]]]
DOC_KINDS = ("transcript", "scene")
# paths -> tags per path (media_tags.tag_clips); runs PyTorch in a separate process
TagsFn = Callable[[list[Path]], dict[str, dict[str, Any]]]


# ------------------------------------------------------------------------------- discovery


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s[:48] or "clip"


def make_clip_id(file: str, sha256: str) -> str:
    """Filename slug + first 8 hex digits of the content hash: stable, unique, and new when the
    file changes. Never contains '::' (chunk ids are `doc_id::cNNNN`)."""
    return f"{slug(Path(file).stem)}-{sha256[:8]}"


def discover(src: Path) -> tuple[list[Path], list[str]]:
    """Media files under `src` (sorted) and the non-media files skipped (reported, not dropped
    silently)."""
    media, skipped = [], []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rel = p.relative_to(src).as_posix()
        if p.suffix.lower() in MEDIA_EXTENSIONS:
            media.append(p)
        elif p.name not in SIDECAR_FILES and p.parent.name != "references":
            skipped.append(rel)
    return media, skipped


def read_vocabulary(path: Path) -> list[str]:
    """Names and terms to bias transcription towards: one per line, `#` starts a comment."""
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.split("#", 1)[0].strip()
        if t and t not in terms:
            terms.append(t)
    return terms


def read_licences(path: Path) -> dict[str, dict[str, str]]:
    """`file,licence,source_url,notes` rows keyed by file (relative to the clip folder)."""
    if not path.exists():
        raise DataError(f"licence file not found: {path} (columns: file,licence,source_url,notes)")
    out = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            f = (row.get("file") or "").strip()
            lic = (row.get("licence") or "").strip()
            if f and lic:
                out[f] = {
                    "licence": lic,
                    "source_url": (row.get("source_url") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
    return out


# ------------------------------------------------------------------------------- transcripts


def keep_segment(s: Segment) -> bool:
    if not s.text.strip():
        return False
    return not (s.no_speech_prob > NO_SPEECH_MAX and s.avg_logprob < LOGPROB_MIN)


def transcript_cache_path(cache_dir: Path, sha256: str, t: Transcriber) -> Path:
    key = stable_hash({"model": t.model, "revision": t.revision, "params": t.params}, 12)
    return cache_dir / "media" / f"{sha256}-{key}.json"


def cached_transcribe(
    clip: Clip, t: Transcriber, cache_dir: Path | None
) -> tuple[Transcript, bool]:
    """Transcribe once per (file content, model, revision, parameters); returns (t, cache_hit)."""
    path = transcript_cache_path(cache_dir, clip.sha256, t) if cache_dir else None
    if path is not None and path.exists():
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8"))), True
    tr = t.transcribe(clip.path)
    if path is not None:
        write_json(path, asdict(tr))
    return tr, False


# ------------------------------------------------------------------------------- windows


def make_windows(
    segments: Sequence[Segment], window_s: float = 60.0, max_s: float = 90.0, overlap: int = 1
) -> list[list[Segment]]:
    """Group consecutive segments into windows of about `window_s` seconds, cut only at segment
    boundaries and never longer than `max_s` (a single longer segment stays whole). The last
    `overlap` segments of a window open the next one, so an answer spanning a cut is still
    retrievable from one window; a tail longer than half a window is not repeated."""
    if window_s <= 0 or max_s < window_s:
        raise ValueError("need 0 < window_s <= max_s")
    segs = sorted(segments, key=lambda s: (s.start, s.end))
    windows: list[list[Segment]] = []
    cur: list[Segment] = []

    def tail(w: list[Segment]) -> list[Segment]:
        t = w[-overlap:] if overlap else []
        return t if t and t[-1].end - t[0].start < window_s / 2 else []

    for seg in segs:
        if cur and seg.end - cur[0].start > max_s:  # adding it would make the window too long
            windows.append(cur)
            t = tail(cur)
            cur = t if t and seg.end - t[0].start <= max_s else []
        cur.append(seg)
        if seg.end - cur[0].start >= window_s:
            windows.append(cur)
            cur = tail(cur)
    if cur and (not windows or cur != windows[-1][-len(cur) :]):  # not just the overlap tail
        windows.append(cur)
    return windows


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def window_doc_id(clip_id: str, start: float, end: float) -> str:
    return f"{clip_id}#t{int(start):06d}-{math.ceil(end):06d}"


def window_documents(clip: Clip, tr: Transcript, windows: list[list[Segment]]) -> list[Document]:
    docs, seen = [], set()
    for w in windows:
        start, end = w[0].start, max(s.end for s in w)
        doc_id = window_doc_id(clip.clip_id, start, end)
        n = 2
        while doc_id in seen:  # identical integer bounds (very short windows)
            doc_id = f"{window_doc_id(clip.clip_id, start, end)}-{n}"
            n += 1
        seen.add(doc_id)
        text = " ".join(f"[{fmt_ts(s.start)}] {s.text.strip()}" for s in w)
        docs.append(
            Document(
                doc_id=doc_id,
                title=f"{clip.file} [{fmt_ts(start)}–{fmt_ts(end)}]",
                text=text,
                source="media",
                metadata={
                    "kind": "transcript",
                    "clip_id": clip.clip_id,
                    "file": clip.file,
                    "start_s": round(start, 2),
                    "end_s": round(end, 2),
                    "language": tr.language,
                    "segments": len(w),
                    "asr_model": f"{tr.model}@{(tr.revision or 'unpinned')[:8]}",
                },
            )
        )
    return docs


def scene_documents(clip: Clip, scenes: list[Any], captioner: str) -> list[Document]:
    docs, seen = [], set()
    for s in scenes:
        if not s.caption.strip():
            continue
        doc_id = f"{clip.clip_id}#s{int(s.start_s):06d}-{math.ceil(s.end_s):06d}"
        while doc_id in seen:
            doc_id += "b"
        seen.add(doc_id)
        span = f"{fmt_ts(s.start_s)}–{fmt_ts(s.end_s)}"
        docs.append(
            Document(
                doc_id=doc_id,
                title=f"{clip.file} [{span}] (scene)",
                text=f"[{span}] Scene: {s.caption.strip()}",
                source="media",
                metadata={
                    "kind": "scene",
                    "clip_id": clip.clip_id,
                    "file": clip.file,
                    "start_s": s.start_s,
                    "end_s": s.end_s,
                    "keyframe_s": s.keyframe_s,
                    "captioner": captioner,
                },
            )
        )
    return docs


def _duration_words(seconds: float) -> str:
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = [f"{h} h"] if h else []
    if h or m:
        parts.append(f"{m} min")
    parts.append(f"{sec} s")
    return " ".join(parts)


def card_document(
    clip: Clip,
    tr: Transcript | None,
    music: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> Document:
    """The clip's technical details as sentences, so they can be retrieved and asked about."""
    info = clip.info
    duration = info.duration_s if info and info.duration_s else (tr.duration if tr else 0.0)
    kind = info.kind if info else "media"
    parts = [f"{clip.file} is {'a video' if kind == 'video' else 'an audio'} clip"]
    parts[0] += f", {_duration_words(duration)} long"
    if info:
        parts[0] += f", {info.container} container"
        if info.video:
            v = info.video
            parts.append(
                f"Video: {v.get('codec')} {v.get('width')}x{v.get('height')}"
                + (f" at {v['fps']:g} fps" if v.get("fps") else "")
            )
        if info.audio:
            a = info.audio
            ch = a.get("channels")
            layout = {1: "mono", 2: "stereo"}.get(ch, f"{ch} channels") if ch else ""
            rate = f"{a['sample_rate'] / 1000:g} kHz" if a.get("sample_rate") else ""
            parts.append(" ".join(x for x in (f"Audio: {a.get('codec')}", rate, layout) if x))
        for k in ("title", "artist", "album", "date", "genre", "comment"):
            if info.tags.get(k):
                parts.append(f"{k.capitalize()} tag: {info.tags[k]}")
    if tr is not None:
        parts.append(
            f"Spoken language: {tr.language} (detected with probability "
            f"{tr.language_probability:.2f})"
            if tr.segments
            else "No speech was detected"
        )
    if music is not None:
        from elevate_rag.data.media_music import music_sentence

        parts.append(music_sentence(music))
    if tags is not None:
        from elevate_rag.data.media_tags import tags_sentence

        parts.append(tags_sentence(tags))
    text = ". ".join(parts) + "."
    return Document(
        doc_id=f"{clip.clip_id}#card",
        title=f"{clip.file} (clip details)",
        text=text,
        source="media",
        metadata={
            "kind": "card",
            "clip_id": clip.clip_id,
            "file": clip.file,
            "duration_s": round(duration, 2),
            "sha256": clip.sha256,
            "licence": clip.licence.get("licence"),
            **({"probe": asdict(info)} if info else {}),
            **({"music": music} if music is not None else {}),
            **({"tags": {k: v for k, v in tags.items() if k != "probs"}} if tags else {}),
        },
    )


# ------------------------------------------------------------------------------- questions


def build_questions(
    rows: Sequence[dict[str, Any]],
    name: str,
    clips: dict[str, Clip],
    docs_by_clip: dict[str, list[Document]],
) -> tuple[list[Question], list[dict[str, Any]]]:
    """Map each question's clip + time span to gold windows; reject (and report) bad rows."""
    questions, rejected = [], []
    for i, row in enumerate(rows):
        try:
            questions.append(_question(i, row, name, clips, docs_by_clip))
        except (DataError, ValueError, TypeError, KeyError) as exc:
            rejected.append({"line": i + 1, "error": str(exc)[:300], "row": row})
    return questions, rejected


def _question(
    i: int,
    row: dict[str, Any],
    name: str,
    clips: dict[str, Clip],
    docs_by_clip: dict[str, list[Document]],
) -> Question:
    text = str(row.get("question") or "").strip()
    if not text:
        raise DataError("empty question")
    qtype = str(row.get("query_type") or "other")
    if qtype not in QUERY_TYPES:
        raise DataError(f"query_type {qtype!r} not in {QUERY_TYPES}")
    answerable = bool(row.get("answerable", True))
    answers = [str(a) for a in row.get("answers") or []]
    if answerable and not answers:
        raise DataError("an answerable question needs `answers`")
    gold: list[str] = []
    refs = row.get("clips") or ([row["clip"]] if row.get("clip") else [])
    spans = row.get("spans") or (
        [{"clip": row["clip"], "start_s": row.get("start_s"), "end_s": row.get("end_s")}]
        if row.get("clip")
        else []
    )
    if answerable and not refs:
        raise DataError("an answerable question needs `clip` (or `clips`/`spans`)")
    for ref in refs:
        if ref not in clips:
            raise DataError(f"unknown clip {ref!r}")
    if row.get("card"):
        gold += [f"{clips[r].clip_id}#card" for r in refs]
    else:
        for sp in spans:
            clip = clips[sp["clip"]]
            lo, hi = sp.get("start_s"), sp.get("end_s")
            kinds = row.get("kinds") or ["transcript"]  # v1 meaning unless asked otherwise
            if bad := set(kinds) - set(DOC_KINDS):
                raise DataError(f"unknown kinds {sorted(bad)}; use {DOC_KINDS}")
            wins = [d for d in docs_by_clip[clip.clip_id] if d.metadata["kind"] in kinds]
            if lo is None or hi is None:
                raise DataError(f"{sp['clip']}: `start_s` and `end_s` are required")
            lo, hi = float(lo), float(hi)
            dur = clip.info.duration_s if clip.info else None
            if hi < lo or lo < 0 or (dur and lo > dur + 1):
                raise DataError(f"{sp['clip']}: time span [{lo}, {hi}] is outside the clip")
            hit = [
                d.doc_id for d in wins if d.metadata["start_s"] < hi and d.metadata["end_s"] > lo
            ]
            if not hit:
                raise DataError(
                    f"{sp['clip']}: no {'/'.join(kinds)} document overlaps [{lo}, {hi}]"
                )
            gold += hit
    evidence = row.get("evidence")
    return Question(
        question_id=str(row.get("id") or f"{name}-q{i:03d}"),
        dataset=name,
        question=text,
        answers=answers,
        query_type=qtype,
        answerable=answerable,
        gold_doc_ids=list(dict.fromkeys(gold)),
        gold_evidence=[str(evidence)] if evidence else [],
        metadata={"clips": list(refs), "spans": spans} if refs else {},
    )


def _tags_params() -> dict[str, Any]:
    from elevate_rag.data.media_tags import CLAP_REPO, CLAP_REVISION, LABELS, TAGS_VERSION

    return {"model": CLAP_REPO, "revision": CLAP_REVISION, "labels": LABELS, "v": TAGS_VERSION}


def read_questions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise DataError(f"{path}:{n}: each line must be a JSON object")
        rows.append(obj)
    return rows


# ------------------------------------------------------------------------------- quality


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Word-level edit distance / reference length (lowercased, punctuation removed)."""

    def words(t: str) -> list[str]:
        return re.sub(r"[^\w\s']", " ", t.lower()).split()

    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


# ------------------------------------------------------------------------------- orchestration


def ingest_media(
    src: Path,
    name: str,
    data_dir: Path,
    transcriber: Transcriber,
    probe: ProbeFn | None,
    *,
    questions_path: Path,
    licences_path: Path | None = None,
    window_s: float = 60.0,
    max_s: float = 90.0,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    music: MusicFn | None = None,
    scenes: ScenesFn | None = None,
    tags: TagsFn | None = None,
) -> dict[str, Any]:
    """Build `data_dir/processed/<name>/` from the clips under `src`; returns the manifest.
    With `music`, every clip card also gets its music details (tempo, key, level, content); with
    `scenes`, every video clip gets one captioned document per detected scene; with `tags` (needs
    `music`), clips labelled music or mixed also get genre / mood / instrument tags."""
    if tags is not None and music is None:
        raise DataError("sound tags need the music analysis (it decides which clips are music)")
    if not src.is_dir():
        raise DataError(f"not a folder: {src}")
    licences = read_licences(licences_path or src / "licences.csv")
    paths, skipped = discover(src)
    if not paths:
        raise DataError(f"no media files under {src} (extensions: {sorted(MEDIA_EXTENSIONS)})")
    rels = [p.relative_to(src).as_posix() for p in paths]
    missing = [r for r in rels if r not in licences]
    if missing:
        raise DataError(
            f"{len(missing)} clip(s) without a licence row in licences.csv, e.g. {missing[:3]}"
        )
    clips: dict[str, Clip] = {}
    transcripts: dict[str, Transcript] = {}
    docs_by_clip: dict[str, list[Document]] = {}
    per_clip: list[dict[str, Any]] = []
    for p in paths:
        rel = p.relative_to(src).as_posix()
        sha = file_sha256(p)
        clip = Clip(p, rel, sha, p.stat().st_size, make_clip_id(rel, sha), licences[rel])
        if progress:
            progress(f"{rel}: probing and transcribing")
        clip.info = probe(p) if probe else None
        tr, hit = cached_transcribe(clip, transcriber, cache_dir)
        transcripts[rel] = tr
        kept = [s for s in tr.segments if keep_segment(s)]
        windows = make_windows(kept, window_s, max_s)
        duration = (clip.info.duration_s if clip.info else 0.0) or tr.duration
        speech_share = sum(s.end - s.start for s in kept) / duration if duration else None
        music_info = music(p, speech_share) if music else None
        docs = [*window_documents(clip, tr, windows), card_document(clip, tr, music_info)]
        is_video = clip.info.kind == "video" if clip.info else p.suffix.lower() in VIDEO_EXTENSIONS
        scene_docs: list[Document] = []
        if scenes is not None and is_video:
            found, scene_settings = scenes(p)
            scene_docs = scene_documents(clip, found, str(scene_settings.get("captioner")))
            docs += scene_docs
        clips[rel], docs_by_clip[clip.clip_id] = clip, docs
        transcript_text = " ".join(s.text for s in kept)
        ref = src / "references" / f"{p.name}.txt"
        per_clip.append(
            {
                "file": rel,
                "clip_id": clip.clip_id,
                "sha256": sha,
                "bytes": clip.bytes,
                "licence": clip.licence,
                "kind": clip.info.kind if clip.info else None,
                "duration_s": clip.info.duration_s if clip.info else tr.duration,
                "language": tr.language,
                "language_probability": round(tr.language_probability, 3),
                "segments": len(tr.segments),
                "segments_dropped": len(tr.segments) - len(kept),
                "windows": len(windows),
                "transcript_cached": hit,
                **({"music": music_info} if music_info is not None else {}),
                **({"scenes": len(scene_docs)} if scenes is not None and is_video else {}),
                "wer_vs_reference": (
                    word_error_rate(ref.read_text(encoding="utf-8"), transcript_text)
                    if ref.exists()
                    else None
                ),
            }
        )
    if tags is not None:
        musical = [
            c for c in per_clip if (c.get("music") or {}).get("content") in ("music", "mixed")
        ]
        tag_results = tags([src / c["file"] for c in musical]) if musical else {}
        for c in musical:
            clip, t = clips[c["file"]], tag_results[str(src / c["file"])]
            c["tags"] = {k: v for k, v in t.items() if k != "probs"}
            docs = docs_by_clip[clip.clip_id]
            i = next(j for j, d in enumerate(docs) if d.metadata["kind"] == "card")
            docs[i] = card_document(clip, transcripts[c["file"]], c["music"], t)
    questions, rejected = build_questions(read_questions(questions_path), name, clips, docs_by_clip)
    documents = [d for docs in docs_by_clip.values() for d in docs]
    notes = [
        f"{len(clips)} clip(s) transcribed with {transcriber.model}",
        *(f"skipped non-media file: {s}" for s in skipped),
    ]
    lic = sorted({c.licence["licence"] for c in clips.values()})
    scene_params = scenes.settings if scenes is not None and hasattr(scenes, "settings") else None
    return write_processed(
        name,
        LoadResult(documents, questions, rejected, notes),
        data_dir / "processed" / name,
        params={
            "loader": MEDIA_LOADER_VERSION,
            "seed": 13,
            "window_s": window_s,
            "max_window_s": max_s,
            "overlap_segments": 1,
            **({"music": "librosa-0.11 tempo/key/level"} if music else {}),
            **({"scenes": scene_params or True} if scenes else {}),
            **({"tags": _tags_params()} if tags else {}),
            "segment_filter": {"no_speech_prob_max": NO_SPEECH_MAX, "avg_logprob_min": LOGPROB_MIN},
            "asr": {
                "model": transcriber.model,
                "revision": transcriber.revision,
                **transcriber.params,
            },
        },
        sources={
            "clips": per_clip,
            "skipped_files": skipped,
            "questions_file": questions_path.name,
        },
        license="; ".join(lic),
        description=f"audio/video clips from {src.name}/ transcribed to timestamped windows",
    )
