"""Genre / mood / instrument tags for `ingest-media --tags` (docs/MEDIA_V2_DESIGN.md §3, v2c).

Zero-shot audio-text matching with CLAP (`laion/clap-htsat-fused`, Apache-2.0, safetensors
weights only, pinned revision). For every label group the clip is compared with one text prompt
per label; a softmax over the group gives each label's probability.

CLAP needs PyTorch, and PyTorch must not share a process with CTranslate2 (OpenMP; see
`utils.guard_openmp`). So `ingest-media` never imports this module's heavy part: `tag_clips`
runs `python -m elevate_rag.data.media_tags` as a **separate process** and reads its JSON output.
Results are cached per (file content, model, revision, labels).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from elevate_rag.utils import ConfigError, file_sha256, stable_hash, write_json

CLAP_REPO = "laion/clap-htsat-fused"
CLAP_REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"  # verified 2026-11-27, Apache-2.0
CLAP_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
CLAP_RATE = 48_000
MAX_SECONDS = 60.0  # longer clips: the first minute is tagged
PROMPT = "This is a sound of {}."
TAGS_VERSION = "tags-v2"
# A closed label list always yields *some* genre with a confident-looking softmax, even for speech
# (a talk was tagged "jazz, 0.88" in testing). So only clips the librosa analysis labels music or
# mixed are tagged (`ingest-media --tags` requires `--music`). CLAP's own content scores were not
# reliable enough to gate on (it rated a synthetic instrumental "noise", a talk "music"); they
# are recorded for inspection only.
CONTENT_LABELS = ["music", "a person speaking", "noise"]
LABELS: dict[str, list[str]] = {
    "content": CONTENT_LABELS,
    "genre": [
        "classical music",
        "rock music",
        "pop music",
        "jazz",
        "electronic music",
        "hip hop",
        "folk music",
        "ambient music",
    ],
    "mood": ["calm music", "energetic music", "happy music", "sad music", "tense music"],
    "instrument": [
        "piano",
        "acoustic guitar",
        "electric guitar",
        "drums",
        "string instruments",
        "synthesizer",
        "singing voice",
        "brass instruments",
    ],
}
MIN_PROB = {"genre": 0.3, "mood": 0.35, "instrument": 0.2}  # below this a group says nothing


def summarise(probs: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Top label per group (if its probability clears MIN_PROB); up to two instruments."""
    out: dict[str, Any] = {}
    for group, ps in probs.items():
        if group == "content":
            continue
        ranked = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        keep = [(lab, p) for lab, p in ranked if p >= MIN_PROB.get(group, 0.3)]
        if group == "instrument":
            out[group] = [{"label": lab, "p": round(p, 2)} for lab, p in keep[:2]]
        elif keep:
            out[group] = {"label": keep[0][0], "p": round(keep[0][1], 2)}
    return out


def tags_sentence(tags: dict[str, Any]) -> str:
    parts = []
    if g := tags.get("genre"):
        parts.append(f"genre {g['label']} ({g['p']:.2f})")
    if m := tags.get("mood"):
        parts.append(f"mood {m['label']} ({m['p']:.2f})")
    if inst := tags.get("instrument"):
        parts.append("instruments " + ", ".join(i["label"] for i in inst))
    return "Sound tags: " + ("; ".join(parts) if parts else "none confident")


def _cache_path(cache_dir: Path, sha: str) -> Path:
    key = stable_hash(
        {"repo": CLAP_REPO, "rev": CLAP_REVISION, "labels": LABELS, "v": TAGS_VERSION}
    )
    return cache_dir / "media" / "tags" / f"{sha}-{key[:12]}.json"


def tag_clips(paths: list[Path], cache_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Tags for each clip (keyed by path string), computed in a separate process."""
    out: dict[str, dict[str, Any]] = {}
    todo: list[Path] = []
    shas = {str(p): file_sha256(p) for p in paths}
    for p in paths:
        cp = _cache_path(cache_dir, shas[str(p)]) if cache_dir else None
        if cp is not None and cp.exists():
            out[str(p)] = json.loads(cp.read_text())
        else:
            todo.append(p)
    if todo:
        proc = subprocess.run(
            [sys.executable, "-m", "elevate_rag.data.media_tags", *map(str, todo)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode != 0:
            raise ConfigError(f"sound tagging failed: {proc.stderr.strip()[-600:]}")
        probs = json.loads(proc.stdout.strip().splitlines()[-1])
        for p in todo:
            tags = {"probs": probs[str(p)], **summarise(probs[str(p)]), "model": CLAP_REPO}
            out[str(p)] = tags
            if cache_dir:
                write_json(_cache_path(cache_dir, shas[str(p)]), tags)
    return out


# ------------------------------------------------------------------------------- worker process


def _worker(paths: list[str]) -> dict[str, dict[str, dict[str, float]]]:  # pragma: no cover
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import ClapModel, ClapProcessor

    from elevate_rag.data.media_music import decode

    local = snapshot_download(CLAP_REPO, revision=CLAP_REVISION, allow_patterns=CLAP_FILES)
    model = ClapModel.from_pretrained(local, use_safetensors=True).eval()
    processor = ClapProcessor.from_pretrained(local)
    texts = {g: [PROMPT.format(lab) for lab in labels] for g, labels in LABELS.items()}

    def emb(out: Any) -> Any:
        # transformers 5 returns an output object (projected, L2-normalised `pooler_output`);
        # older versions return the projected tensor itself.
        x = out.pooler_output if hasattr(out, "pooler_output") else out
        return x / x.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        text_emb = {
            g: emb(model.get_text_features(**processor(text=t, return_tensors="pt", padding=True)))
            for g, t in texts.items()
        }
        result = {}
        for p in paths:
            y = decode(Path(p), CLAP_RATE)[: int(MAX_SECONDS * CLAP_RATE)]
            if y.size == 0:
                y = np.zeros(CLAP_RATE, dtype=np.float32)
            inputs = processor(audio=[y], sampling_rate=CLAP_RATE, return_tensors="pt")
            a = emb(model.get_audio_features(**inputs))
            groups = {}
            for g, tn in text_emb.items():
                logits = model.logit_scale_a.exp() * (a @ tn.T)
                pr = logits.softmax(dim=-1)[0].tolist()
                groups[g] = {lab: round(float(x), 4) for lab, x in zip(LABELS[g], pr, strict=True)}
            result[p] = groups
    return result


if __name__ == "__main__":  # pragma: no cover - run as a separate process by tag_clips
    print(json.dumps(_worker(sys.argv[1:])))
