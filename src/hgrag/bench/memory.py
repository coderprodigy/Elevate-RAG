"""Memory probes for the benchmarks: Docker container usage, or this process's peak RSS."""

from __future__ import annotations

import re
import resource
import subprocess
import sys

_UNITS = {
    "b": 1 / 2**20,
    "kib": 1 / 1024,
    "mib": 1.0,
    "gib": 1024.0,
    "kb": 1 / 1000,
    "mb": 1.0,
    "gb": 1000.0,
}
_USAGE = re.compile(r"([\d.]+)\s*([KMG]i?B|B)", re.IGNORECASE)


def parse_docker_mem(usage: str) -> float | None:
    """'1.234GiB / 4GiB' → 1263.6 (MB). None if unparseable."""
    match = _USAGE.search(usage)
    if match is None:
        return None
    return round(float(match.group(1)) * _UNITS[match.group(2).lower()], 1)


def docker_memory_mb(container: str) -> float | None:
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_docker_mem(result.stdout) if result.returncode == 0 else None


def process_peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 1024, 1)  # bytes vs KiB


CONTAINERS = {
    "qdrant": "hgrag-qdrant",
    "neo4j": "hgrag-neo4j",
    "milvus": "hgrag-milvus",
    "pgvector": "hgrag-pgvector",
    "memgraph": "hgrag-memgraph",
    "postgres": "hgrag-postgres-age",
}
