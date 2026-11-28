"""Real media backends (extra `media`): PyAV probing and faster-whisper transcription.

Skipped without the extra. The transcription test also needs the macOS `say` command (to make a
speech clip with a known transcript) and the pinned `small` model (downloaded once, ~0.5 GB).
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("faster_whisper")

from elevate_rag.data.media import word_error_rate
from elevate_rag.data.media_backends import WhisperTranscriber, probe
from elevate_rag.utils import ConfigError, DataError

pytestmark = pytest.mark.integration


def _tone(path: Path, seconds: float = 2.0, rate: int = 16_000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def test_probe_reads_technical_details(tmp_path: Path) -> None:
    p = tmp_path / "tone.wav"
    _tone(p)
    info = probe(p)
    assert info.kind == "audio" and info.container == "wav"
    assert info.duration_s == pytest.approx(2.0, abs=0.05)
    assert info.audio == {"codec": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    (tmp_path / "bad.mp4").write_text("not a video")
    with pytest.raises(DataError, match="not a readable"):
        probe(tmp_path / "bad.mp4")


def test_unknown_model_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="unknown ASR model"):
        WhisperTranscriber("tiny-made-up")
    t = WhisperTranscriber("org/model@abc")
    assert (t.model, t.revision) == ("org/model", "abc")
    assert t.params["temperature"] == 0.0 and t.params["compute_type"] in ("int8", "float16")


@pytest.mark.network
@pytest.mark.skipif(shutil.which("say") is None, reason="needs the macOS `say` command")
def test_whisper_small_transcribes_known_speech(tmp_path: Path) -> None:
    text = "The Velmora Observatory was founded in 1998 by Ada Brennan."
    p = tmp_path / "speech.wav"
    subprocess.run(["say", "-o", str(p), "--data-format=LEI16@16000", text], check=True)
    tr = WhisperTranscriber("small", language="en").transcribe(p)
    hyp = " ".join(s.text for s in tr.segments)
    assert tr.language == "en" and tr.segments
    wer = word_error_rate(text, hyp)
    assert wer is not None and wer <= 0.2
    assert tr.segments[0].start < 1.0 and tr.segments[-1].end <= tr.duration + 0.5
