"""Video scenes for `ingest-media --scenes` (docs/MEDIA_V2_DESIGN.md §2, extra `media`).

* `detect_scenes`: cuts where consecutive sampled frames (64x36 colour) differ by more than a
  threshold; scenes shorter than `min_scene_s` are merged, longer than `max_scene_s` are split.
  PyAV + numpy only.
* `keyframe_jpeg`: the frame nearest a scene's middle, as JPEG (Pillow).
* `VisionCaptioner`: describes a keyframe through an OpenAI-compatible vision endpoint (vLLM,
  Ollama) with its own small client and the project's response cache. It does not use the cost
  ledger, so it accepts local, unpriced servers only. Captions are deterministic (temperature 0,
  seed 0) and cached by (model, prompt version, image bytes).
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from elevate_rag.cache import KVCache
from elevate_rag.utils import ConfigError, DataError, ProviderError

PROMPT_VERSION = "scene-v1"
SCENE_PROMPT = (
    "Describe what is visible in this video frame in 1-3 short factual sentences: people, "
    "objects, actions and any readable text (quote it exactly). Do not guess names or identities."
)


@dataclass
class Scene:
    start_s: float
    end_s: float
    keyframe_s: float
    caption: str = ""


def _av() -> Any:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("scene detection needs the extra: uv sync --extra media") from exc
    return av


def sample_frames(path: Path, sample_fps: float = 4.0) -> tuple[list[float], np.ndarray, float]:
    """(times, frames as N x 36 x 64 x 3 colour floats 0-1, duration) for the first video stream.
    Colour, not grey: cuts between scenes of similar brightness but different colour are common."""
    av = _av()
    times: list[float] = []
    frames: list[np.ndarray] = []
    next_t = 0.0
    try:
        container = av.open(str(path))
    except Exception as exc:
        raise DataError(f"{path.name}: not a readable video ({exc})") from exc
    with container:
        if not container.streams.video:
            return [], np.zeros((0, 36, 64, 3)), 0.0
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < next_t:
                continue
            small = frame.reformat(width=64, height=36, format="rgb24")
            frames.append(small.to_ndarray().astype(np.float32) / 255.0)
            times.append(float(frame.time))
            next_t = frame.time + 1.0 / sample_fps
        duration = max(duration, times[-1] if times else 0.0)
    return times, np.stack(frames) if frames else np.zeros((0, 36, 64, 3)), duration


def cut_points(
    times: list[float], frames: np.ndarray, threshold: float = 0.12, min_scene_s: float = 2.0
) -> list[float]:
    """Times where the mean absolute difference to the previous sample exceeds `threshold`,
    at least `min_scene_s` apart."""
    cuts: list[float] = []
    for i in range(1, len(times)):
        diff = float(np.abs(frames[i] - frames[i - 1]).mean())
        if diff > threshold and times[i] - (cuts[-1] if cuts else 0.0) >= min_scene_s:
            cuts.append(times[i])
    return cuts


def scenes_from_cuts(cuts: list[float], duration: float, max_scene_s: float = 30.0) -> list[Scene]:
    bounds = [0.0, *cuts, duration]
    scenes: list[Scene] = []
    for a, b in pairwise(bounds):
        if b - a <= 0:
            continue
        n = max(1, int(np.ceil((b - a) / max_scene_s)))
        step = (b - a) / n
        for k in range(n):
            s, e = a + k * step, a + (k + 1) * step
            scenes.append(Scene(round(s, 2), round(e, 2), round((s + e) / 2, 2)))
    return scenes


def detect_scenes(
    path: Path,
    *,
    threshold: float = 0.12,
    min_scene_s: float = 2.0,
    max_scene_s: float = 30.0,
    sample_fps: float = 4.0,
) -> list[Scene]:
    times, frames, duration = sample_frames(path, sample_fps)
    if not times:
        return []
    return scenes_from_cuts(
        cut_points(times, frames, threshold, min_scene_s), duration, max_scene_s
    )


def keyframe_jpeg(path: Path, t: float, max_side: int = 768) -> bytes:
    """JPEG of the first frame at or after `t` seconds (longest side at most `max_side`)."""
    av = _av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        best = None
        for frame in container.decode(stream):
            best = frame
            if frame.time is not None and frame.time >= t:
                break
        if best is None:
            raise DataError(f"{path.name}: no video frame at {t:.1f} s")
        img = best.to_image()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class VisionCaptioner:
    """Captions images through an OpenAI-compatible vision endpoint (local, unpriced servers)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        cache_dir: Path | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 160,
    ) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env) if api_key_env else None
        self.timeout_s, self.max_tokens = timeout_s, max_tokens
        self.cache = KVCache(cache_dir, "captions") if cache_dir else None
        self.calls = 0

    def caption(self, jpeg: bytes) -> str:
        key = KVCache.key(
            {"model": self.model, "prompt": PROMPT_VERSION, "image": KVCache.key(jpeg.hex())}
        )
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return str(hit["text"])
        import httpx

        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCENE_PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers,
                timeout=self.timeout_s,
            )  # fmt: skip
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"captioner {self.model}: {exc}"[:300]) from exc
        self.calls += 1
        text = " ".join(text.split())
        if self.cache is not None:
            self.cache.set(key, {"text": text})
        return text


def scene_fn(
    captioner: VisionCaptioner, **detect_kw: Any
) -> Callable[[Path], tuple[list[Scene], dict[str, Any]]]:
    """For `media.ingest_media(scenes=...)`: detect, keyframe and caption the scenes of a clip."""

    settings = {"captioner": captioner.model, "prompt_version": PROMPT_VERSION, **detect_kw}

    def run(path: Path) -> tuple[list[Scene], dict[str, Any]]:
        scenes = detect_scenes(path, **detect_kw)
        for s in scenes:
            s.caption = captioner.caption(keyframe_jpeg(path, s.keyframe_s))
        return scenes, settings

    run.settings = settings  # type: ignore[attr-defined]  # recorded in the dataset manifest
    return run
