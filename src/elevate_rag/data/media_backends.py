"""Heavy media backends, imported only by `data ingest-media` (extra `media`).

* `probe`: technical details of a clip with PyAV (it bundles FFmpeg, so no system ffmpeg).
* `WhisperTranscriber`: faster-whisper with a pinned CTranslate2 Whisper model and fixed,
  deterministic decoding settings (docs/MEDIA_INGESTION_DESIGN.md §4).

Never imports PyTorch or FAISS: CTranslate2 and ONNX Runtime bring their own OpenMP runtime, and
on macOS a second copy next to those would abort the process (see `utils.guard_openmp`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from elevate_rag.data.media import VIDEO_EXTENSIONS, ClipInfo, Segment, Transcript
from elevate_rag.utils import ConfigError, DataError

# Verified on Hugging Face 2026-11-27: ungated, MIT (CTranslate2 conversions of OpenAI Whisper).
WHISPER_MODELS = {
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
}
DECODE = {
    "beam_size": 5,
    "temperature": 0.0,
    "vad_filter": True,
    "condition_on_previous_text": False,
    "word_timestamps": False,
}
TAG_KEYS = ("title", "artist", "album", "date", "genre", "comment")


def _codec(name: str) -> str:
    """FFmpeg decoder name → codec name ("mp3float" → "mp3")."""
    return name.removesuffix("float")


def _container(format_name: str, suffix: str) -> str:
    """FFmpeg demuxer names are lists ("mov,mp4,m4a,…"): pick the one matching the file."""
    names = format_name.split(",")
    ext = suffix.lower().lstrip(".")
    return ext if ext in names else names[0]


def _import_av() -> Any:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ConfigError("media ingestion needs the extra: uv sync --extra media") from exc
    return av


def probe(path: Path) -> ClipInfo:
    av = _import_av()
    try:
        container = av.open(str(path))
    except Exception as exc:
        raise DataError(f"{path.name}: not a readable audio/video file ({exc})") from exc
    with container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        audio = video = None
        if container.streams.audio:
            a = container.streams.audio[0]
            audio = {
                "codec": _codec(a.codec_context.name),
                "sample_rate": a.codec_context.sample_rate,
                "channels": a.codec_context.channels,
            }
            if not duration and a.duration and a.time_base:
                duration = float(a.duration * a.time_base)
        if container.streams.video:
            v = container.streams.video[0]
            fps = v.average_rate
            video = {
                "codec": _codec(v.codec_context.name),
                "width": v.codec_context.width,
                "height": v.codec_context.height,
                "fps": round(float(fps), 3) if fps else None,
            }
        tags = {k: str(container.metadata[k])[:200] for k in TAG_KEYS if container.metadata.get(k)}
        kind = "video" if video and path.suffix.lower() in VIDEO_EXTENSIONS else "audio"
        return ClipInfo(
            kind=kind,
            duration_s=round(duration, 3),
            container=_container(container.format.name, path.suffix),
            audio=audio,
            video=video,
            tags=tags,
        )


class WhisperTranscriber:
    """faster-whisper with a pinned model. `model` is a key of WHISPER_MODELS or `repo@revision`."""

    def __init__(
        self,
        model: str = "small",
        *,
        language: str | None = None,
        device: str = "auto",
        cache_dir: Path | None = None,
        vocabulary: list[str] | None = None,
    ) -> None:
        if model in WHISPER_MODELS:
            repo, rev = WHISPER_MODELS[model]
        elif "@" in model:
            repo, rev = model.split("@", 1)
        else:
            raise ConfigError(
                f"unknown ASR model {model!r}: use one of {sorted(WHISPER_MODELS)} or repo@revision"
            )
        self.model: str = repo
        self.revision: str | None = rev
        self.device = device
        self.cache_dir = cache_dir
        self.language = language
        self._model: Any = None
        self._device, self._compute_type = self._resolved_device()
        # int8 on CPU and float16 on GPU give slightly different transcripts: part of the
        # cache key and the dataset manifest.
        # Rare names and jargon (people, places, products) are misheard when the model has never
        # seen them; faster-whisper's `hotwords` biases every decoding window towards them.
        self.hotwords = ", ".join(vocabulary) if vocabulary else None
        self.params: dict[str, Any] = {
            **DECODE,
            "language": language,
            "hotwords": self.hotwords,
            "device_class": self._device,
            "compute_type": self._compute_type,
        }

    def _resolved_device(self) -> tuple[str, str]:
        try:
            import ctranslate2
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ConfigError("media ingestion needs the extra: uv sync --extra media") from exc

        device = self.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        return device, ("float16" if device == "cuda" else "int8")

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel, download_model
            except ImportError as exc:  # pragma: no cover - depends on the optional extra
                raise ConfigError("media ingestion needs the extra: uv sync --extra media") from exc
            hf = self.cache_dir or (
                Path(os.environ["HF_HOME"]) / "hub" if "HF_HOME" in os.environ else None
            )
            path = download_model(
                self.model,
                revision=self.revision,
                cache_dir=str(hf) if hf else None,
                local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
            )
            self._model = WhisperModel(path, device=self._device, compute_type=self._compute_type)
        return self._model

    def transcribe(self, path: Path) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(
            str(path), language=self.language, hotwords=self.hotwords, **DECODE
        )
        segs = [
            Segment(
                start=round(float(s.start), 2),
                end=round(float(s.end), 2),
                text=s.text.strip(),
                avg_logprob=round(float(s.avg_logprob), 4),
                no_speech_prob=round(float(s.no_speech_prob), 4),
            )
            for s in segments  # a generator: decoding happens while iterating
        ]
        return Transcript(
            language=info.language,
            language_probability=float(info.language_probability),
            duration=float(info.duration),
            segments=segs,
            model=self.model,
            revision=self.revision,
            params=self.params,
        )
