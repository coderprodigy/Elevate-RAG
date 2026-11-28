"""Music details for `ingest-media --music` (docs/MEDIA_V2_DESIGN.md §3, extra `music`).

Tempo, key, loudness, brightness and a speech / music / mixed / silence label, from librosa 0.11
on audio decoded with PyAV (so MP3 / MP4 work without a system ffmpeg).

Numba (used by librosa) is forced to its `workqueue` threading layer *before* librosa is
imported: its OpenMP layer could load a second OpenMP runtime next to CTranslate2's, the macOS
abort `utils.guard_openmp` exists to prevent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from elevate_rag.utils import ConfigError

RATE = 22_050
# Krumhansl–Kessler key profiles (major, minor), starting at the tonic.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SILENCE_DBFS = -50.0  # peak below this: silence
PULSE_MIN = 0.3  # periodicity of the onset envelope above which a clip has a steady beat
SPEECH_SHARE_MIN = 0.3  # share of the clip Whisper's voice-activity filter kept as speech
KEY_CONFIDENCE_MIN = 0.5  # below this (e.g. drums only) the key is reported as unclear
# Calibrated on synthetic signals (tests/integration/test_media_music.py): speech 0.15 pulse,
# beats 0.84-0.94. A quiet beat under louder speech scored 0.27 and is reported as speech:
# conservative by design.


def _librosa() -> Any:
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("music details need the extra: uv sync --extra music") from exc
    return librosa


def decode(path: Path, rate: int = RATE) -> np.ndarray:
    """Mono float32 samples at `rate`, decoded with PyAV alone (importing faster-whisper would
    load CTranslate2, which the sound-tagging worker must keep away from PyTorch)."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("audio decoding needs the extra: uv sync --extra media") from exc
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(container.streams.audio[0]):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    """Krumhansl–Schmuckler: correlate the mean chroma with the 24 rotated key profiles."""
    best, score = "unknown", -1.0
    if not np.any(chroma_mean > 0):
        return best, 0.0
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1])
            if r > score:
                best, score = f"{NOTES[tonic]} {mode}", r
    return best, score


def pulse_clarity(onset_env: np.ndarray, sr: int, hop: int) -> float:
    """Normalised autocorrelation peak of the onset envelope within 40–220 BPM (0–1)."""
    x = onset_env - onset_env.mean()
    if len(x) < 4 or not np.any(x):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac /= ac[0]
    lo = max(1, int(60 / 220 * sr / hop))
    hi = min(len(ac) - 1, int(60 / 40 * sr / hop))
    return float(max(0.0, ac[lo : hi + 1].max())) if hi > lo else 0.0


def classify(peak_dbfs: float, pulse: float, speech_share: float | None) -> str:
    if peak_dbfs < SILENCE_DBFS:
        return "silence"
    speechy = speech_share is not None and speech_share >= SPEECH_SHARE_MIN
    if pulse >= PULSE_MIN:
        return "mixed" if speechy else "music"
    return "speech" if speechy else "other"


def analyse(path: Path, speech_share: float | None = None) -> dict[str, Any]:
    """Music details of one clip. `speech_share`: share of the clip that holds speech (from the
    transcript), used to tell music from speech over music."""
    librosa = _librosa()
    y = decode(path)
    if y.size == 0:
        return {"content": "silence", "duration_s": 0.0}
    hop = 512
    peak = float(np.max(np.abs(y)))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else -120.0
    out: dict[str, Any] = {
        "duration_s": round(len(y) / RATE, 2),
        "peak_dbfs": round(float(peak_dbfs), 1),
        "rms_dbfs_mean": round(float(20 * np.log10(max(float(rms.mean()), 1e-6))), 1),
        "speech_share": None if speech_share is None else round(speech_share, 2),
    }
    if peak_dbfs < SILENCE_DBFS:
        out["content"] = "silence"
        return out
    onset_env = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=hop)
    pulse = pulse_clarity(onset_env, RATE, hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=RATE, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=RATE, hop_length=hop)
    key, key_conf = estimate_key(chroma.mean(axis=1))
    if key_conf < KEY_CONFIDENCE_MIN:
        key = "unclear"
    centroid = librosa.feature.spectral_centroid(y=y, sr=RATE, hop_length=hop)[0]
    out.update(
        {
            "content": classify(float(peak_dbfs), pulse, speech_share),
            "pulse_clarity": round(pulse, 2),
            "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
            "beats": len(beats),
            "key": key,
            "key_confidence": round(key_conf, 2),
            "spectral_centroid_hz": round(float(centroid.mean())),
        }
    )
    return out


def music_sentence(info: dict[str, Any]) -> str:
    """One sentence for the clip card."""
    content = info.get("content", "other")
    if content not in ("music", "mixed"):
        return f"Music: none detected (content: {content})"
    lead = "Music" if content == "music" else "Music under speech"
    key = (
        f"key {info['key']} (confidence {info['key_confidence']:.2f})"
        if info["key"] != "unclear"
        else "key unclear"
    )
    return (
        f"{lead}: about {info['tempo_bpm']:g} BPM, {key}, mean level "
        f"{info['rms_dbfs_mean']:g} dBFS, brightness {info['spectral_centroid_hz']} Hz"
    )
