"""CLAP sound tags in their separate worker process (needs torch, transformers and the media
extra; downloads the pinned model, ~615 MB, on first use)."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

from elevate_rag.data.media_tags import LABELS, tag_clips

# find_spec, not importorskip: importing torch into the test process would make later FAISS tests
# trip the macOS OpenMP guard. The worker imports torch in its own process.
_missing = [m for m in ("torch", "transformers", "av") if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.skipif(bool(_missing), reason=f"needs {_missing}"),
]


def test_worker_returns_probabilities_per_group_and_caches(tmp_path: Path) -> None:
    rate = 48_000
    t = np.arange(4 * rate) / rate
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    p = tmp_path / "tone.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    out = tag_clips([p], cache_dir=tmp_path / "cache")[str(p)]
    for group, labels in LABELS.items():
        assert set(out["probs"][group]) == set(labels)
        assert abs(sum(out["probs"][group].values()) - 1.0) < 1e-3
    assert out["model"] == "laion/clap-htsat-fused"
    assert tag_clips([p], cache_dir=tmp_path / "cache")[str(p)] == out  # served from the cache
