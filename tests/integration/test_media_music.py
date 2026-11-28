"""Music details on real decoded audio (extras `media` + `music`): tempo, key, content label, and
that librosa's numba never brings a second OpenMP runtime next to CTranslate2's."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("faster_whisper")

from elevate_rag.data.media_music import RATE, analyse

pytestmark = pytest.mark.integration


def _wav(path: Path, y: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _clicks(bpm: float, secs: float = 20.0) -> np.ndarray:
    y = np.zeros(int(secs * RATE))
    t = np.arange(int(0.02 * RATE)) / RATE
    click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)
    for b in np.arange(0, secs, 60 / bpm):
        i = int(b * RATE)
        seg = y[i : i + len(click)]
        seg += click[: len(seg)]
    return 0.8 * y


def _chords(chords: list[list[int]], reps: int = 4) -> np.ndarray:
    out = []
    t = np.arange(RATE) / RATE
    env = np.minimum(1, np.minimum(t / 0.02, (1 - t) / 0.05))
    for _ in range(reps):
        for notes in chords:
            freqs = [440 * 2 ** ((m - 69) / 12) for m in notes]
            tone = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) for f in freqs)
            out.append(0.5 * tone * env / len(freqs))
    return np.concatenate(out)


@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_tempo_of_click_tracks(tmp_path: Path, bpm: int) -> None:
    info = analyse(_wav(tmp_path / f"c{bpm}.wav", _clicks(bpm)), speech_share=0.0)
    assert info["content"] == "music" and abs(info["tempo_bpm"] - bpm) <= 3
    assert info["key"] == "unclear"  # clicks have no key


@pytest.mark.parametrize(
    ("chords", "key"),
    [
        ([[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]], "A minor"),  # Am Dm E Am
        ([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]], "C major"),  # C F G C
    ],
)
def test_key_of_chord_progressions(tmp_path: Path, chords: list[list[int]], key: str) -> None:
    info = analyse(_wav(tmp_path / "k.wav", _chords(chords)), speech_share=0.0)
    assert info["key"] == key and info["key_confidence"] > 0.8


def test_silence_and_speech_labels(tmp_path: Path) -> None:
    assert analyse(_wav(tmp_path / "s.wav", np.zeros(5 * RATE)))["content"] == "silence"
    rng = np.random.default_rng(0)  # noise bursts at irregular times: no steady beat
    y = np.zeros(15 * RATE)
    for start in rng.uniform(0, 14, 25):
        i = int(start * RATE)
        y[i : i + RATE // 5] += rng.normal(0, 0.2, RATE // 5)[: len(y[i : i + RATE // 5])]
    assert analyse(_wav(tmp_path / "n.wav", y), speech_share=0.8)["content"] == "speech"


def test_a_single_openmp_runtime_with_ctranslate2_and_librosa(tmp_path: Path) -> None:
    clip = _wav(tmp_path / "c.wav", _clicks(120, 5))
    code = f"""
import sys, ctranslate2
from elevate_rag.data.media_music import analyse
analyse(__import__("pathlib").Path({str(clip)!r}))
libs = []
if sys.platform == "darwin":
    import ctypes
    dyld = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    libs = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
else:
    libs = [l.split()[-1] for l in open("/proc/self/maps") if len(l.split()) > 5]
import re
names = {{l.rsplit("/", 1)[-1] for l in libs}}
omp = sorted(n for n in names if re.match(r"lib[ig]?omp[0-9]*[.](dylib|so)", n))
print(omp)
print(len(omp) <= 1)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "True", out.stdout
