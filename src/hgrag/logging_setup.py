"""Console logging with rich formatting."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Vendor HTTP clients are noisy at INFO; keep them to warnings.
    for noisy in ("httpx", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
