"""Download every Hugging Face model a pipeline config needs, at its pinned revision.

Run once on a machine with internet (e.g. a cluster login node); compute nodes then run with
HF_HUB_OFFLINE=1 from the same HF_HOME. API models (Anthropic, OpenAI) and Ollama tags are
skipped; models served by vLLM are downloaded so the server can start offline.

    uv run python scripts/download_models.py configs/pipelines/cluster_gpu.yaml
    uv run python scripts/download_models.py configs/pipelines/cluster_gpu.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

from elevate_rag.config import load_pipeline_config

HF_PROVIDERS = {"sentence-transformers", "cross-encoder", "openai-compatible"}
# Weights + tokenizer + configs; skips duplicate formats (.bin/.pth/.onnx/.gguf) to save space.
PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
    "*.jinja",
    "1_Pooling/*",
    "2_Normalize/*",
    "sentencepiece*",
]


def wanted(cfg_path: str) -> dict[str, str | None]:
    os.environ.setdefault("ELEVATE_GEN_URL", "unused")  # configs may name endpoints; not needed
    os.environ.setdefault("ELEVATE_JUDGE_URL", "unused")
    os.environ.setdefault("OPENAI_COMPAT_BASE_URL", "unused")
    cfg = load_pipeline_config(cfg_path)
    out: dict[str, str | None] = {}
    for role, m in vars(cfg.models).items():
        if m is None or m.provider not in HF_PROVIDERS or "/" not in m.model:
            continue  # API model, Ollama tag (no "/"), or offline stand-in
        if m.revision is None:
            sys.exit(f"{role}: {m.model} has no pinned revision; pin it in the config first")
        out[m.model] = m.revision
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    models: dict[str, str | None] = {}
    for c in args.configs:
        models.update(wanted(c))
    if not models:
        print("no Hugging Face models to download")
        return
    from huggingface_hub import snapshot_download

    for repo, rev in sorted(models.items()):
        print(f"{repo}@{rev}", flush=True)
        if not args.dry_run:
            path = snapshot_download(repo, revision=rev, allow_patterns=PATTERNS)
            print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
