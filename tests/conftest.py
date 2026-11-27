from pathlib import Path

import pytest

from elevate_rag.config import Settings
from elevate_rag.datasets import fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_path(tmp_path):
    destination = tmp_path / "corpus"
    fixture(ROOT / "tests" / "fixtures", destination)
    return destination


@pytest.fixture
def settings():
    return Settings.read(ROOT / "configs" / "synthetic.toml")
