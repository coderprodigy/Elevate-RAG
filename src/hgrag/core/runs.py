"""Run identity: who ran what, and under which run ID. Recorded with every ledger row."""

from __future__ import annotations

import getpass
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


def new_run_id(prefix: str = "adhoc") -> str:
    """Sortable, unique run ID, e.g. ``e1-20261124T201500Z-3f9a1c``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def current_user() -> str:
    """The git user name if configured, otherwise the OS login name."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return getpass.getuser()
    name = result.stdout.strip()
    return name or getpass.getuser()


@dataclass(frozen=True)
class RunContext:
    """Labels attached to every recorded API call, so spend can be grouped later."""

    run_id: str = field(default_factory=new_run_id)
    experiment: str = "adhoc"
    phase: str = "dev"
    user: str = field(default_factory=current_user)
