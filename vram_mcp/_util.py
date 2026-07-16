"""Small shared helpers with no dependencies on any sibling module."""

from __future__ import annotations

import subprocess
from typing import Optional

BYTES_PER_MB = 1024 * 1024


def bytes_to_mb(value, default=None) -> Optional[int]:
    """Best-effort bytes → whole MB; ``default`` on missing/garbage input.

    Callers pick the fallback that matches their contract: ``0`` where a
    number is always expected (Ollama sizes), ``None`` where "unreported"
    is meaningful (NVML per-process sizes on Windows/WDDM).
    """
    try:
        return int(value) // BYTES_PER_MB
    except (TypeError, ValueError):
        return default


def run_capture(cmd: list[str], timeout: int) -> Optional[str]:
    """Run ``cmd`` and return stdout, or ``None`` on ANY failure (missing
    binary, timeout, non-zero exit). For callers whose contract is
    best-effort telemetry — never raises."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        # FileNotFoundError is an OSError subclass — one tuple covers both.
        return None
    return result.stdout
