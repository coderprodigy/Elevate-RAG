"""Architecture rule: only hgrag.llm.providers may import vendor LLM SDKs (CLAUDE.md rule 1)."""

import ast
from pathlib import Path

from hgrag.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "hgrag"
PROVIDERS_DIR = SRC / "llm" / "providers"
VENDOR_SDKS = {"anthropic", "openai"}


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_vendor_sdks_only_imported_by_providers() -> None:
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in SRC.rglob("*.py")
        if not path.is_relative_to(PROVIDERS_DIR)
        and _imported_top_level_modules(path) & VENDOR_SDKS
    ]
    assert offenders == [], f"route these through hgrag.llm.gateway instead: {offenders}"
