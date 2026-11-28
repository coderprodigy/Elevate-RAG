"""Audio/video ingestion core (data/media.py), with a fake transcriber: runs without the extra."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from elevate_rag.data.media import (
    ClipInfo,
    Segment,
    Transcript,
    card_document,
    discover,
    fmt_ts,
    ingest_media,
    keep_segment,
    make_clip_id,
    make_windows,
    word_error_rate,
)
from elevate_rag.data.prepare import DatasetStore
from elevate_rag.utils import DataError


def _segs(*spans: tuple[float, float]) -> list[Segment]:
    return [Segment(s, e, f"seg {i}") for i, (s, e) in enumerate(spans)]


# ------------------------------------------------------------------------------- windows


def test_windows_cut_at_segment_boundaries_with_one_segment_overlap() -> None:
    segs = _segs((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
    w = make_windows(segs, window_s=60, max_s=90)
    assert [[s.text for s in x] for x in w] == [
        ["seg 0", "seg 1", "seg 2"],  # reaches 60 s → closed
        ["seg 2", "seg 3", "seg 4"],  # opens with the overlap segment
    ]
    assert all(x[-1].end - x[0].start <= 90 for x in w)


def test_windows_respect_the_hard_maximum_and_keep_long_segments_whole() -> None:
    w = make_windows(_segs((0, 50), (50, 120), (120, 125)), window_s=60, max_s=90)
    # seg 1 (70 s) closes its own window and is too long to repeat as overlap
    assert [[s.text for s in x] for x in w] == [["seg 0"], ["seg 1"], ["seg 2"]]
    assert make_windows(_segs((0, 200)), 60, 90)[0][0].end == 200  # one long segment
    assert make_windows([], 60, 90) == []
    assert len(make_windows(_segs((0, 10)), 60, 90)) == 1  # short clip: one window
    with pytest.raises(ValueError):
        make_windows([], window_s=60, max_s=30)


def test_no_window_is_only_the_overlap_tail() -> None:
    w = make_windows(_segs((0, 30), (30, 61)), window_s=60, max_s=90)
    assert len(w) == 1  # the closing segment would otherwise reappear alone


def test_hallucination_filter_and_formatting() -> None:
    assert keep_segment(Segment(0, 1, "hello", avg_logprob=-0.3, no_speech_prob=0.9))
    assert not keep_segment(Segment(0, 1, "thanks for watching", -1.5, 0.9))
    assert not keep_segment(Segment(0, 1, "  "))
    assert fmt_ts(61.9) == "01:01" and fmt_ts(3725) == "1:02:05"
    cid = make_clip_id("Talks/My Interview.MP4", "a" * 64)
    assert cid == "my-interview-aaaaaaaa" and "::" not in cid


def test_vocabulary_file(tmp_path: Path) -> None:
    from elevate_rag.data.media import read_vocabulary

    (tmp_path / "v.txt").write_text("# names\nKorrin Hill\n\nDornholt  # a town\nKorrin Hill\n")
    assert read_vocabulary(tmp_path / "v.txt") == ["Korrin Hill", "Dornholt"]


def test_word_error_rate() -> None:
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cat sat", "the bat sat down") == pytest.approx(2 / 3)
    assert word_error_rate("", "x") is None


def test_card_describes_the_clip() -> None:
    from elevate_rag.data.media import Clip

    clip = Clip(Path("v.mp4"), "v.mp4", "b" * 64, 10, "v-bbbbbbbb", {"licence": "CC0"})
    clip.info = ClipInfo(
        kind="video",
        duration_s=751.2,
        container="mp4",
        audio={"codec": "aac", "sample_rate": 48000, "channels": 2},
        video={"codec": "h264", "width": 1920, "height": 1080, "fps": 25.0},
        tags={"title": "Lab history"},
    )
    doc = card_document(clip, Transcript("en", 0.99, 751.2, [Segment(0, 1, "hi")], "m", "r"))
    assert doc.doc_id == "v-bbbbbbbb#card"
    for part in ("a video clip", "12 min 31 s", "h264 1920x1080 at 25 fps", "48 kHz stereo"):
        assert part in doc.text
    assert "Title tag: Lab history" in doc.text and doc.metadata["licence"] == "CC0"


# ------------------------------------------------------------------------------- ingestion


class FakeTranscriber:
    model, revision = "fake/asr", "r1"
    params: dict[str, Any] = {"beam_size": 1}

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, path: Path) -> Transcript:
        self.calls += 1
        if "silent" in path.name:
            return Transcript("en", 0.5, 10.0, [], self.model, self.revision, self.params)
        segs = [
            Segment(0.0, 20.0, "The lab was founded in 1998 by Ada Brennan."),
            Segment(20.0, 45.0, "It moved to Korrin Hill in 2004."),
            Segment(45.0, 70.0, "Thanks for watching!", avg_logprob=-2.0, no_speech_prob=0.95),
            Segment(70.0, 95.0, "Tomas Ferrel took over in 2019."),
        ]
        return Transcript("en", 0.99, 95.0, segs, self.model, self.revision, self.params)


def _probe(path: Path) -> ClipInfo:
    return ClipInfo(kind="audio", duration_s=95.0, container="wav")


def _folder(tmp: Path, licences: bool = True) -> Path:
    src = tmp / "clips"
    src.mkdir()
    (src / "talk.wav").write_bytes(b"RIFF fake audio")
    (src / "silent.mp3").write_bytes(b"ID3 fake audio")
    (src / "notes.txt").write_text("not media")
    if licences:
        (src / "licences.csv").write_text(
            "file,licence,source_url,notes\ntalk.wav,CC-BY-4.0,https://x,\nsilent.mp3,CC0,,\n"
        )
    rows: list[dict[str, Any]] = [
        {"question": "When was the lab founded?", "answers": ["1998"], "clip": "talk.wav",
         "start_s": 5, "end_s": 15, "query_type": "single_hop", "evidence": "founded in 1998"},
        {"question": "Who took over?", "answers": ["Tomas Ferrel"], "clip": "talk.wav",
         "start_s": 71, "end_s": 80},
        {"question": "How long is talk.wav?", "answers": ["1 min 35 s"], "clip": "talk.wav",
         "card": True},
        {"question": "Who was the first mayor?", "answers": [], "answerable": False,
         "query_type": "null"},
        {"question": "Bad clip", "answers": ["x"], "clip": "missing.wav", "start_s": 0,
         "end_s": 1},
        {"question": "Outside", "answers": ["x"], "clip": "talk.wav", "start_s": 500,
         "end_s": 510},
    ]  # fmt: skip
    (src / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return src


def test_ingest_builds_a_normal_dataset(tmp_path: Path) -> None:
    src, fake = _folder(tmp_path), FakeTranscriber()
    m = ingest_media(
        src,
        "media_t",
        tmp_path / "data",
        fake,
        _probe,
        questions_path=src / "questions.jsonl",
        window_s=40,
        max_s=60,
        cache_dir=tmp_path / "cache",
    )
    ds = DatasetStore.open(tmp_path / "data", "media_t")
    ds.verify()
    docs = {d.doc_id: d for d in ds.documents()}
    windows = [d for d in docs.values() if d.metadata["kind"] == "transcript"]
    assert len([d for d in docs.values() if d.metadata["kind"] == "card"]) == 2
    assert all("Thanks for watching" not in d.text for d in windows)  # hallucination dropped
    first = min(windows, key=lambda d: d.metadata["start_s"])
    assert first.text.startswith("[00:00] The lab was founded") and "[00:20]" in first.text
    assert first.title.startswith("talk.wav [00:00–")
    qs = {q.question: q for q in ds.questions()}
    assert len(qs) == 4 and m["counts"]["rejected_records"] == 2  # bad clip + outside span
    assert qs["When was the lab founded?"].gold_doc_ids == [first.doc_id]
    assert qs["How long is talk.wav?"].gold_doc_ids[0].endswith("#card")
    assert not qs["Who was the first mayor?"].answerable
    assert m["evidence_verbatim_rate"] == 1.0
    clip = {c["file"]: c for c in m["sources"]["clips"]}
    assert clip["talk.wav"]["segments_dropped"] == 1 and clip["silent.mp3"]["windows"] == 0
    assert m["sources"]["skipped_files"] == ["notes.txt"]
    assert m["params"]["asr"]["model"] == "fake/asr" and "CC-BY-4.0" in m["license"]
    # re-ingesting hits the transcript cache: no new transcription
    ingest_media(src, "media_t", tmp_path / "data", fake, _probe,
                 questions_path=src / "questions.jsonl", window_s=40, max_s=60,
                 cache_dir=tmp_path / "cache")  # fmt: skip
    assert fake.calls == 2


def test_ingest_refuses_clips_without_a_licence(tmp_path: Path) -> None:
    src = _folder(tmp_path, licences=False)
    with pytest.raises(DataError, match="licence file not found"):
        ingest_media(src, "m", tmp_path / "d", FakeTranscriber(), _probe,
                     questions_path=src / "questions.jsonl")  # fmt: skip
    (src / "licences.csv").write_text("file,licence\ntalk.wav,CC0\n")
    with pytest.raises(DataError, match="without a licence"):
        ingest_media(src, "m", tmp_path / "d", FakeTranscriber(), _probe,
                     questions_path=src / "questions.jsonl")  # fmt: skip


def test_discover_skips_sidecars_and_hidden_files(tmp_path: Path) -> None:
    for name in ("a.wav", "b.MP4", ".hidden.wav", "licences.csv", "questions.jsonl", "x.pdf"):
        (tmp_path / name).write_text("x")
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "a.wav.txt").write_text("ref")
    media, skipped = discover(tmp_path)
    assert [p.name for p in media] == ["a.wav", "b.MP4"] and skipped == ["x.pdf"]


def test_media_core_never_loads_torch_or_faiss() -> None:
    code = (
        "import sys, elevate_rag.data.media, elevate_rag.data.media_backends;"
        "print(any(m in sys.modules for m in ('torch', 'faiss')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"  # macOS OpenMP clash stays impossible


# ------------------------------------------------------------------------------- music (pure parts)


def test_key_estimation_and_classification_rules() -> None:
    import numpy as np

    from elevate_rag.data.media_music import (
        _MAJOR,
        _MINOR,
        classify,
        estimate_key,
        music_sentence,
        pulse_clarity,
    )

    assert estimate_key(np.roll(_MAJOR, 7))[0] == "G major"  # profile itself, rotated to G
    assert estimate_key(np.roll(_MINOR, 9))[0] == "A minor"
    assert estimate_key(np.zeros(12)) == ("unknown", 0.0)
    assert classify(-60, 0.9, None) == "silence"
    assert classify(-10, 0.9, 0.0) == "music" and classify(-10, 0.9, 0.8) == "mixed"
    assert classify(-10, 0.1, 0.8) == "speech" and classify(-10, 0.1, None) == "other"
    env = np.tile([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 40)
    assert pulse_clarity(env, 22050, 512) > 0.8  # a click every 21 frames ≈ 123 BPM
    assert pulse_clarity(np.zeros(100), 22050, 512) == 0.0
    info = {"content": "music", "tempo_bpm": 120.0, "key": "A minor", "key_confidence": 0.9,
            "rms_dbfs_mean": -14.0, "spectral_centroid_hz": 1800}  # fmt: skip
    assert music_sentence(info).startswith("Music: about 120 BPM, key A minor (confidence 0.90)")
    assert "key unclear" in music_sentence({**info, "key": "unclear"})
    assert music_sentence({"content": "speech"}) == "Music: none detected (content: speech)"


def test_music_details_reach_the_card_and_are_off_by_default(tmp_path: Path) -> None:
    src = _folder(tmp_path)
    calls: list[tuple[str, float | None]] = []

    def fake_music(path: Path, share: float | None) -> dict[str, Any]:
        calls.append((path.name, share))
        return {"content": "mixed", "tempo_bpm": 96.0, "key": "D major", "key_confidence": 0.8,
                "rms_dbfs_mean": -20.0, "spectral_centroid_hz": 1500}  # fmt: skip

    kw: dict[str, Any] = {"questions_path": src / "questions.jsonl", "window_s": 40, "max_s": 60}
    plain = ingest_media(src, "m", tmp_path / "d1", FakeTranscriber(), _probe, **kw)
    again = ingest_media(src, "m", tmp_path / "d2", FakeTranscriber(), _probe, **kw)
    assert plain["version"] == again["version"]  # deterministic; no music fields without the flag
    with_music = ingest_media(
        src, "m", tmp_path / "d3", FakeTranscriber(), _probe, music=fake_music, **kw
    )
    assert with_music["version"] != plain["version"]
    card = next(
        d for d in DatasetStore.open(tmp_path / "d3", "m").documents()
        if d.doc_id.endswith("#card") and d.metadata["file"] == "talk.wav"
    )  # fmt: skip
    assert "Music under speech: about 96 BPM, key D major" in card.text
    assert card.metadata["music"]["tempo_bpm"] == 96.0
    share = dict(calls)["talk.wav"]
    assert share is not None and 0.7 < share < 0.75  # 70 s of kept speech in a 95 s clip


# ------------------------------------------------------------------------------- scenes


def test_cut_points_and_scene_splitting() -> None:
    import numpy as np

    from elevate_rag.data.media_vision import cut_points, scenes_from_cuts

    times = [i * 0.25 for i in range(40)]  # 10 s sampled at 4 fps
    frames = np.zeros((40, 36, 64, 3), dtype=np.float32)
    frames[12:] = 0.5  # cut at 3.0 s
    frames[13] = 0.52  # small flicker: not a cut
    frames[14:] = 0.5
    frames[16:] = 0.1  # cut at 4.0 s: too close to the previous one (min 2 s)
    frames[32:] = 0.9  # cut at 8.0 s
    assert cut_points(times, frames, threshold=0.12, min_scene_s=2.0) == [3.0, 8.0]
    scenes = scenes_from_cuts([3.0, 8.0], 10.0, max_scene_s=30)
    assert [(s.start_s, s.end_s, s.keyframe_s) for s in scenes] == [
        (0.0, 3.0, 1.5), (3.0, 8.0, 5.5), (8.0, 10.0, 9.0)
    ]  # fmt: skip
    long = scenes_from_cuts([], 70.0, max_scene_s=30)  # one long shot is split in 3
    assert [(s.start_s, s.end_s) for s in long] == [(0.0, 23.33), (23.33, 46.67), (46.67, 70.0)]


def test_scene_documents_and_scene_gold(tmp_path: Path) -> None:
    from elevate_rag.data.media_vision import Scene

    src = _folder(tmp_path)
    (src / "tour.mp4").write_bytes(b"fake video")
    with (src / "licences.csv").open("a") as fh:
        fh.write("tour.mp4,CC0,,\n")
    first = [json.loads(x) for x in (src / "questions.jsonl").read_text().splitlines() if x]
    rows = [
        *first[:2],
        {"question": "What is written on the sign?", "answers": ["OPEN"], "clip": "tour.mp4",
         "start_s": 12, "end_s": 14, "kinds": ["scene"]},
        {"question": "Bad kind", "answers": ["x"], "clip": "tour.mp4", "start_s": 0,
         "end_s": 1, "kinds": ["ocr"]},
    ]  # fmt: skip
    (src / "q.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def fake_scenes(path: Path) -> tuple[list[Any], dict[str, Any]]:
        return [Scene(0.0, 10.0, 5.0, "A lab bench."), Scene(10.0, 20.0, 15.0, "A sign: OPEN."),
                Scene(20.0, 25.0, 22.5, "  ")], {"captioner": "fake-vlm"}  # fmt: skip

    def probe(path: Path) -> ClipInfo:
        kind = "video" if path.suffix == ".mp4" else "audio"
        return ClipInfo(kind=kind, duration_s=95.0, container=path.suffix[1:])

    m = ingest_media(src, "ms", tmp_path / "d", FakeTranscriber(), probe,
                     questions_path=src / "q.jsonl", window_s=40, max_s=60,
                     scenes=fake_scenes)  # fmt: skip
    docs = {d.doc_id: d for d in DatasetStore.open(tmp_path / "d", "ms").documents()}
    scene_docs = [d for d in docs.values() if d.metadata["kind"] == "scene"]
    assert len(scene_docs) == 2  # the empty caption is skipped
    sign = next(d for d in scene_docs if "OPEN" in d.text)
    assert sign.text == "[00:10–00:20] Scene: A sign: OPEN." and sign.title.endswith("(scene)")
    assert all(d.metadata["file"] == "tour.mp4" for d in scene_docs)  # audio clips: no scenes
    qs = {q.question: q for q in DatasetStore.open(tmp_path / "d", "ms").questions()}
    assert qs["What is written on the sign?"].gold_doc_ids == [sign.doc_id]
    first = qs["When was the lab founded?"].gold_doc_ids  # default kinds: transcript only
    assert first and all(docs[g].metadata["kind"] == "transcript" for g in first)
    assert m["counts"]["rejected_records"] == 1  # unknown kind "ocr"
    clip = {c["file"]: c for c in m["sources"]["clips"]}
    assert clip["tour.mp4"]["scenes"] == 2 and "scenes" not in clip["talk.wav"]


def test_vision_captioner_request_and_cache(tmp_path: Path) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from elevate_rag.data.media_vision import VisionCaptioner

    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = json.dumps({"choices": [{"message": {"content": " A red   sign. "}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a: Any) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cap = VisionCaptioner("vlm", f"http://127.0.0.1:{srv.server_port}/v1", cache_dir=tmp_path)
        assert cap.caption(b"\xff\xd8jpeg") == "A red sign."
        assert cap.caption(b"\xff\xd8jpeg") == "A red sign." and cap.calls == 1  # cached
        assert len(seen) == 1
        msg = seen[0]["messages"][0]["content"]
        assert seen[0]["temperature"] == 0 and seen[0]["model"] == "vlm"
        assert msg[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        cap.caption(b"\xff\xd8other")
        assert len(seen) == 2  # a different image is a new request
    finally:
        srv.shutdown()


# ------------------------------------------------------------------------------- sound tags


def test_tag_summary_thresholds() -> None:
    from elevate_rag.data.media_tags import summarise, tags_sentence

    probs = {
        "content": {"music": 0.9, "a person speaking": 0.05, "noise": 0.05},
        "genre": {"jazz": 0.7, "rock music": 0.3},
        "mood": {"calm music": 0.3, "sad music": 0.3, "happy music": 0.4},
        "instrument": {"piano": 0.5, "drums": 0.3, "synthesizer": 0.15, "singing voice": 0.05},
    }
    t = summarise(probs)
    assert t["genre"] == {"label": "jazz", "p": 0.7} and t["mood"]["label"] == "happy music"
    assert [i["label"] for i in t["instrument"]] == ["piano", "drums"]  # >= 0.2, at most 2
    assert "content" not in t
    assert tags_sentence(t) == (
        "Sound tags: genre jazz (0.70); mood happy music (0.40); instruments piano, drums"
    )
    flat = summarise({"genre": {"jazz": 0.2, "pop music": 0.2}, "mood": {}, "instrument": {}})
    assert "genre" not in flat and tags_sentence(flat) == "Sound tags: none confident"


def test_tags_only_for_clips_labelled_music(tmp_path: Path) -> None:
    src = _folder(tmp_path)
    asked: list[list[str]] = []

    def music(path: Path, share: float | None) -> dict[str, Any]:
        content = "music" if path.name == "silent.mp3" else "speech"
        return {"content": content, "tempo_bpm": 120.0, "key": "C major", "key_confidence": 0.9,
                "rms_dbfs_mean": -15.0, "spectral_centroid_hz": 900}  # fmt: skip

    def tags(paths: list[Path]) -> dict[str, dict[str, Any]]:
        asked.append([p.name for p in paths])
        return {str(p): {"probs": {}, "genre": {"label": "jazz", "p": 0.8}} for p in paths}

    kw: dict[str, Any] = {"questions_path": src / "questions.jsonl", "window_s": 40, "max_s": 60}
    m = ingest_media(
        src, "mt", tmp_path / "d", FakeTranscriber(), _probe, music=music, tags=tags, **kw
    )
    assert asked == [["silent.mp3"]]  # the speech clip is never tagged
    cards = {
        d.metadata["file"]: d
        for d in DatasetStore.open(tmp_path / "d", "mt").documents()
        if d.metadata["kind"] == "card"
    }
    assert "Sound tags: genre jazz (0.80)" in cards["silent.mp3"].text
    assert "Sound tags" not in cards["talk.wav"].text
    assert m["params"]["tags"]["model"] == "laion/clap-htsat-fused"
    with pytest.raises(DataError, match="need the music analysis"):
        ingest_media(src, "mt2", tmp_path / "d", FakeTranscriber(), _probe, tags=tags, **kw)
