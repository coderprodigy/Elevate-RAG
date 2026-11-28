"""Scene detection and keyframes on a real encoded video (extra `media`: PyAV, Pillow)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from elevate_rag.data.media_vision import detect_scenes, keyframe_jpeg  # imports av lazily

av = pytest.importorskip("av")

pytestmark = pytest.mark.integration


def _video(path: Path, colours: list[tuple[int, int, int]], secs_each: float = 3.0) -> None:
    fps = 10
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = 320, 180, "yuv420p"
        i = 0
        for colour in colours:
            for _ in range(int(secs_each * fps)):
                img = np.zeros((180, 320, 3), dtype=np.uint8)
                img[:] = colour
                img[80:100, 140:180] = 255  # same small object in every scene
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, fps)
                i += 1
                for pkt in v.encode(frame):
                    c.mux(pkt)
        for pkt in v.encode(None):
            c.mux(pkt)


def test_cuts_found_within_half_a_second(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    # similar brightness, different colour: only a colour comparison sees these cuts
    _video(p, [(20, 40, 110), (150, 70, 20), (20, 120, 60), (110, 30, 130)])
    scenes = detect_scenes(p)
    starts = [s.start_s for s in scenes][1:]
    assert len(scenes) == 4
    assert all(abs(a - b) <= 0.5 for a, b in zip(starts, [3.0, 6.0, 9.0], strict=True))
    jpeg = keyframe_jpeg(p, scenes[1].keyframe_s)
    assert jpeg[:2] == b"\xff\xd8" and len(jpeg) > 500


def test_audio_only_file_has_no_scenes(tmp_path: Path) -> None:
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    assert detect_scenes(p) == []
