"""NVIDIA GPU VRAM inspection via ``nvidia-smi``.

Pure module: no ``mcp`` import, so it is unit-testable on its own. Degrades
gracefully to an empty result when ``nvidia-smi`` is missing or fails, which is
the "VRAM unknown" signal the rest of the package understands.
"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

# The fields we request, in order, from nvidia-smi.
_QUERY_FIELDS = "index,name,memory.total,memory.used,memory.free"

_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    f"--query-gpu={_QUERY_FIELDS}",
    "--format=csv,noheader,nounits",
]


def _run(cmd: list[str], timeout: int = 5) -> str:
    """Run ``cmd`` and return stdout. Raises on non-zero exit / missing binary.

    Isolated in its own function so tests can monkeypatch it (or callers can
    pass their own ``runner`` to :func:`gpu_status`).
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def _parse_line(line: str) -> Optional[dict]:
    """Parse one CSV row into a GPU dict, or ``None`` if it is malformed."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    index, name, total, used, free = parts
    return {
        "index": int(index),
        "name": name,
        "total_mb": int(total),
        "used_mb": int(used),
        "free_mb": int(free),
    }


def gpu_status(runner: Optional[Callable[..., str]] = None) -> list[dict]:
    """Return per-GPU VRAM info, or ``[]`` if unavailable.

    Each entry: ``{"index", "name", "total_mb", "used_mb", "free_mb"}``.

    Returns ``[]`` on any failure — ``nvidia-smi`` absent (FileNotFoundError),
    timeout, non-zero exit, or an unparseable line — which the caller reads as
    "VRAM unknown" while model operations still work.

    ``runner`` is an injectable ``cmd -> stdout`` callable for testing; defaults
    to the real subprocess-backed :func:`_run`.
    """
    run = runner or _run
    try:
        out = run(_NVIDIA_SMI_CMD)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        OSError,
    ):
        return []

    gpus: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = _parse_line(line)
        except (ValueError, TypeError):
            return []
        if parsed is None:
            return []
        gpus.append(parsed)
    return gpus


def max_free_mb(gpus: list[dict]) -> Optional[int]:
    """Max ``free_mb`` across ``gpus``; ``None`` when the list is empty."""
    if not gpus:
        return None
    return max(g["free_mb"] for g in gpus)
