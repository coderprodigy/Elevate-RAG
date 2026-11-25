"""Download raw dataset files into ``data/raw/<raw_name>/``. Skips files already present."""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from hgrag.data.loaders import DatasetSpec, HFFile

logger = logging.getLogger(__name__)


def hf_token() -> str | Literal[False]:
    """``HF_TOKEN`` from the environment, else anonymous access.

    The token cached by ``huggingface-cli login`` is deliberately ignored: an expired one makes
    even public downloads fail with HTTP 401.
    """
    return os.environ.get("HF_TOKEN", "").strip() or False


def download_dataset(spec: DatasetSpec, raw_root: Path) -> list[Path]:
    """Fetch every source file of ``spec``; returns the local paths."""
    target_dir = raw_root / spec.raw_name
    paths: list[Path] = []
    for source in spec.sources:
        target = target_dir / source.path
        if target.is_file():
            logger.info("%s: already have %s", spec.name, source.path)
        elif isinstance(source, HFFile):
            logger.info("%s: downloading %s from %s", spec.name, source.path, source.repo)
            hf_hub_download(
                source.repo,
                source.path,
                repo_type="dataset",
                local_dir=target_dir,
                token=hf_token(),
            )
        else:
            logger.info("%s: downloading %s", spec.name, source.url)
            _download_url(source.url, target)
        paths.append(target)
    return paths


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)  # only complete files get the final name
