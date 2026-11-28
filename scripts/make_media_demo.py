"""Generate a small SYNTHETIC audio/video demo set for `elevate-rag data ingest-media` (macOS).

Each sentence is spoken separately with the macOS `say` command and the pieces are joined with
short pauses, so the exact time span of every fact is known. That gives a question set with
correct gold time spans and reference transcripts (for word error rate). The facts are fictional
(the same invented town as the synthetic text corpus), so no model can answer from memory.

    uv run --extra media python scripts/make_media_demo.py data/raw/media_demo

Writes 3 clips (WAV audio, MP3 audio, MP4 video), licences.csv, questions.jsonl and
references/<clip>.txt. Not research data: use it to check the pipeline end to end. On Linux,
bring your own clips (docs/MEDIA_INGESTION_DESIGN.md §2, §6).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

RATE = 16_000
PAUSE_S = 0.6

CLIPS: dict[str, list[str]] = {
    "observatory_history.wav": [
        "Welcome to this short history of the Velmora Observatory.",
        "The observatory was founded in nineteen ninety eight by the astronomer Ada Brennan.",
        "It stands on Korrin Hill, four kilometres north of the town of Velmora.",
        "Its first telescope was a refractor with a lens of sixty centimetres.",
        "In two thousand and four the observatory discovered a comet that was named Brennan Seven.",
        "Ada Brennan directed the observatory for twenty one years.",
        "She retired in two thousand and nineteen, and her student Tomas Ferrel took over.",
        "Today the observatory runs public night tours every Friday.",
    ],
    "telescope_tour.mp4": [
        "This is a video tour of the main telescope at the Velmora Observatory.",
        "The current telescope was installed in twenty sixteen.",
        "It is a reflecting telescope with a mirror of one point two metres.",
        "The mirror was ground in the workshop of the glassmaker Ilse Varga in Dornholt.",
        "Tomas Ferrel chose the site of the new dome himself.",
        "The dome turns once every four minutes when tracking the sky.",
    ],
    "river_festival.mp3": [
        "Every September the town of Velmora holds a river festival.",
        "The festival began in nineteen seventy two as a boat race.",
        "The winning crew receives a silver oar made in Dornholt.",
        "Since two thousand and ten the observatory has opened its dome during the festival night.",
    ],
}
# (question, answers, clip, sentence indexes or "card", query_type)
QUESTIONS: list[dict[str, object]] = [
    {
        "q": "In which year was the Velmora Observatory founded?",
        "a": ["1998"],
        "clip": "observatory_history.wav",
        "s": [1],
        "type": "single_hop",
        "evidence": "founded in 1998",
    },
    {
        "q": "Who founded the Velmora Observatory?",
        "a": ["Ada Brennan"],
        "clip": "observatory_history.wav",
        "s": [1],
        "type": "single_hop",
    },
    {
        "q": "On which hill does the observatory stand?",
        "a": ["Korrin Hill"],
        "clip": "observatory_history.wav",
        "s": [2],
        "type": "single_hop",
    },
    {
        "q": "What was the comet discovered by the observatory named?",
        "a": ["Brennan Seven"],
        "clip": "observatory_history.wav",
        "s": [4],
        "type": "single_hop",
    },
    {
        "q": "Who took over the observatory when Ada Brennan retired?",
        "a": ["Tomas Ferrel"],
        "clip": "observatory_history.wav",
        "s": [6],
        "type": "single_hop",
    },
    {
        "q": "How large is the mirror of the current telescope?",
        "a": ["1.2 metres", "1.2 meters", "1.2 m"],
        "clip": "telescope_tour.mp4",
        "s": [2],
        "type": "single_hop",
    },
    {
        "q": "In which town was the telescope mirror ground?",
        "a": ["Dornholt"],
        "clip": "telescope_tour.mp4",
        "s": [3],
        "type": "single_hop",
    },
    {
        "q": "Who chose the site of the new dome, and whose student was he?",
        "a": ["Tomas Ferrel, Ada Brennan's student"],
        "spans": [("telescope_tour.mp4", [4]), ("observatory_history.wav", [6])],
        "type": "multi_hop",
    },
    {
        "q": "Which town made both the festival's silver oar and the telescope mirror?",
        "a": ["Dornholt"],
        "spans": [("river_festival.mp3", [2]), ("telescope_tour.mp4", [3])],
        "type": "multi_hop",
    },
    {
        "q": "Which is older, the river festival or the observatory?",
        "a": ["the river festival"],
        "spans": [("river_festival.mp3", [1]), ("observatory_history.wav", [1])],
        "type": "comparison",
    },
    {
        "q": "Which of the demo clips is a video?",
        "a": ["telescope_tour.mp4"],
        "clip": "telescope_tour.mp4",
        "card": True,
        "type": "single_hop",
    },
    {"q": "Who was the first mayor of Velmora?", "a": [], "answerable": False, "type": "null"},
]
# Need `ingest-media --scenes --music`: the answer is only in the picture (SCENE_CARDS) or the
# music (THEME_FILE), never in speech.
THEME_FILE = "festival_theme.wav"  # C major chords over a 120 BPM beat, no speech
SCENE_QUESTIONS: list[dict[str, object]] = [
    {
        "q": "What text is shown on screen while the narrator describes the mirror's size?",
        "a": ["MIRROR 1.2 M", "Mirror 1.2 m"],
        "clip": "telescope_tour.mp4",
        "s": [2],
        "type": "single_hop",
        "kinds": ["scene"],
    },
    {
        "q": "Which words appear on screen in the scene about the glassmaker's workshop?",
        "a": ["DORNHOLT WORKSHOP", "Dornholt Workshop"],
        "clip": "telescope_tour.mp4",
        "s": [3],
        "type": "single_hop",
        "kinds": ["scene"],
    },
    {
        "q": "In which musical key is the festival theme tune?",
        "a": ["C major"],
        "clip": THEME_FILE,
        "card": True,
        "type": "single_hop",
    },
    {
        "q": "About how many beats per minute is the festival theme tune?",
        "a": ["120", "120 BPM"],
        "clip": THEME_FILE,
        "card": True,
        "type": "single_hop",
    },
]


def theme_tune(secs_per_chord: float = 1.0, reps: int = 4, bpm: float = 120.0) -> np.ndarray:
    """C F G C chords with a click on every beat: key C major, 120 BPM, no speech."""
    chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]  # MIDI notes
    t = np.arange(int(secs_per_chord * RATE)) / RATE
    env = np.minimum(1, np.minimum(t / 0.02, (secs_per_chord - t) / 0.05))
    parts = []
    for _ in range(reps):
        for notes in chords:
            freqs = [440 * 2 ** ((m - 69) / 12) for m in notes]
            tone = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) for f in freqs)
            parts.append(0.4 * tone * env / len(freqs))
    y = np.concatenate(parts)
    ct = np.arange(int(0.02 * RATE)) / RATE
    click = 0.6 * np.sin(2 * np.pi * 1000 * ct) * np.exp(-ct * 200)
    for b in np.arange(0, len(y) / RATE, 60 / bpm):
        i = int(b * RATE)
        seg = y[i : i + len(click)]
        seg += click[: len(seg)]
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


# Reference transcripts use the written form a transcriber produces ("1998", not "nineteen
# ninety eight"), so word error rate measures recognition errors, not number formatting.
WRITTEN = {
    "nineteen ninety eight": "1998",
    "nineteen seventy two": "1972",
    "two thousand and four": "2004",
    "two thousand and nineteen": "2019",
    "two thousand and ten": "2010",
    "twenty sixteen": "2016",
    "twenty one": "21",
    "sixty": "60",
    "one point two": "1.2",
}


def written(text: str) -> str:
    for spoken, digits in WRITTEN.items():
        text = text.replace(spoken, digits)
    return text


def speak(text: str, tmp: Path) -> np.ndarray:
    out = tmp / "s.wav"
    subprocess.run(
        ["say", "-o", str(out), f"--data-format=LEI16@{RATE}", text], check=True, timeout=120
    )
    with wave.open(str(out)) as w:
        if w.getframerate() != RATE or w.getnchannels() != 1:
            raise RuntimeError("unexpected `say` output format")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def build(sentences: list[str], tmp: Path) -> tuple[np.ndarray, list[tuple[float, float]]]:
    pause = np.zeros(int(PAUSE_S * RATE), dtype=np.int16)
    parts, spans, t = [pause], [], PAUSE_S
    for s in sentences:
        audio = speak(s, tmp)
        spans.append((round(t, 2), round(t + len(audio) / RATE, 2)))
        parts += [audio, pause]
        t += len(audio) / RATE + PAUSE_S
    return np.concatenate(parts), spans


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def _encode_audio(container, stream, samples: np.ndarray, fmt: str) -> None:  # type: ignore[no-untyped-def]
    import av

    n = stream.codec_context.frame_size or 1024
    data = samples.astype(np.float32) / 32768.0 if fmt == "fltp" else samples
    pts = 0
    for i in range(0, len(data), n):
        block = data[i : i + n]
        if len(block) < n:
            block = np.pad(block, (0, n - len(block)))
        frame = av.AudioFrame.from_ndarray(block.reshape(1, -1), format=fmt, layout="mono")
        frame.sample_rate, frame.pts = RATE, pts
        pts += n
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)


def write_mp3(path: Path, samples: np.ndarray) -> None:
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libmp3lame", rate=RATE, layout="mono")
        s.codec_context.format = "s16p"
        _encode_audio(c, s, samples, "s16p")


# One scene per sentence of the video: a background colour and a large word, so scene detection
# (cuts at the sentence starts) and captions (the word should be quoted) can be checked.
SCENE_CARDS = [
    ((20, 40, 110), "TELESCOPE TOUR"),
    ((150, 70, 20), "INSTALLED 2016"),
    ((20, 120, 60), "MIRROR 1.2 M"),
    ((160, 150, 30), "DORNHOLT WORKSHOP"),
    ((110, 30, 130), "NEW DOME"),
    ((30, 130, 150), "DOME ROTATION"),
]


def _scene_frame(colour: tuple[int, int, int], word: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 360), colour)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    w = draw.textlength(word, font=font)
    draw.text(((640 - w) / 2, 150), word, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def write_mp4(path: Path, samples: np.ndarray, spans: list[tuple[float, float]]) -> None:
    """H.264 video (one scene card per sentence, 640x360 at 10 fps) + AAC audio."""
    import av

    fps, dur = 10, len(samples) / RATE
    cards = [_scene_frame(c, w) for c, w in SCENE_CARDS]
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = 640, 360, "yuv420p"
        a = c.add_stream("aac", rate=RATE, layout="mono")
        starts = [s - PAUSE_S / 2 for s, _ in spans]  # cut in the pause before each sentence
        for i in range(int(dur * fps) + 1):
            t = i / fps
            k = max([j for j, s in enumerate(starts) if s <= t] or [0])
            frame = av.VideoFrame.from_ndarray(cards[k % len(cards)], format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for pkt in v.encode(frame):
                c.mux(pkt)
        for pkt in v.encode(None):
            c.mux(pkt)
        _encode_audio(c, a, samples, "fltp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    if not shutil.which("say"):
        sys.exit("this generator needs the macOS `say` command; on Linux bring your own clips")
    out: Path = args.out
    (out / "references").mkdir(parents=True, exist_ok=True)
    timing: dict[str, list[tuple[float, float]]] = {}
    with tempfile.TemporaryDirectory() as td:
        for name, sentences in CLIPS.items():
            samples, spans = build(sentences, Path(td))
            timing[name] = spans
            path = out / name
            if name.endswith(".wav"):
                write_wav(path, samples)
            elif name.endswith(".mp3"):
                write_mp3(path, samples)
            else:
                write_mp4(path, samples, spans)
            (out / "references" / f"{name}.txt").write_text(written(" ".join(sentences)) + "\n")
            print(f"{name}: {len(samples) / RATE:.1f} s, {len(sentences)} sentences")
        write_wav(out / THEME_FILE, theme_tune())
        print(f"{THEME_FILE}: 16.0 s, instrumental (C major, 120 BPM)")
    # Invented names a speech model cannot know: the vocabulary hint for `ingest-media`.
    (out / "vocabulary.txt").write_text(
        "# names in the demo clips (one per line)\n"
        "Velmora\nKorrin Hill\nAda Brennan\nBrennan Seven\nTomas Ferrel\nIlse Varga\nDornholt\n"
    )
    lic = "generated locally with macOS `say` (Apple system voice); synthetic, not redistributed"
    with (out / "licences.csv").open("w") as fh:
        fh.write("file,licence,source_url,notes\n")
        for name in CLIPS:
            fh.write(f'{name},"{lic}",,"fictional facts, scripts/make_media_demo.py"\n')
        fh.write(
            f'{THEME_FILE},"synthesised by scripts/make_media_demo.py (CC0)",,"instrumental"\n'
        )

    def rows(questions: list[dict[str, object]], start: int) -> list[dict[str, object]]:
        out_rows = []
        for i, q in enumerate(questions, start=start):
            row: dict[str, object] = {
                "id": f"media-demo-q{i:02d}",
                "question": q["q"],
                "answers": q["a"],
                "query_type": q["type"],
            }
            if q.get("answerable") is False:
                row["answerable"] = False
            elif q.get("card"):
                row.update(clip=q["clip"], card=True)
            else:
                pairs = q.get("spans") or [(q["clip"], q["s"])]
                spans = []
                for clip, idx in pairs:  # type: ignore[union-attr]
                    s, e = timing[clip][min(idx)][0], timing[clip][max(idx)][1]
                    spans.append({"clip": clip, "start_s": s, "end_s": e})
                if len(spans) == 1:
                    row.update(
                        clip=spans[0]["clip"], start_s=spans[0]["start_s"], end_s=spans[0]["end_s"]
                    )
                else:
                    row.update(clips=[sp["clip"] for sp in spans], spans=spans)
            for k in ("evidence", "kinds"):
                if q.get(k):
                    row[k] = q[k]
            out_rows.append(row)
        return out_rows

    base = rows(QUESTIONS, 0)
    scene = rows(SCENE_QUESTIONS, len(QUESTIONS))
    for fname, data in (("questions.jsonl", base), ("questions_v2.jsonl", base + scene)):
        with (out / fname).open("w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in data)
    print(
        f"wrote {out} ({len(QUESTIONS)} questions; +{len(SCENE_QUESTIONS)} picture/music questions)"
    )


if __name__ == "__main__":
    main()
