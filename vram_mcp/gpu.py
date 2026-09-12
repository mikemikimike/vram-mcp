"""NVIDIA GPU VRAM inspection via ``nvidia-smi``.

The observation API keeps an unavailable collector distinct from a successful
empty reading.  The legacy :func:`gpu_status` wrapper remains for callers that
still use ``[]`` as their unknown sentinel.
"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

from .observations import Observation

# The fields we request, in order, from nvidia-smi.
_QUERY_FIELDS = "index,uuid,name,memory.total,memory.used,memory.free"

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
    if len(parts) != 6:
        return None
    index, uuid, name, total, used, free = parts
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "total_mb": int(total),
        "used_mb": int(used),
        "free_mb": int(free),
    }


def observe_gpu(
    index: int = 0,
    runner: Optional[Callable[..., str]] = None,
) -> Observation[list[dict]]:
    """Observe VRAM for one explicitly selected GPU.

    A successful observation contains zero or one row.  A command or parse
    failure has ``data=None`` so callers never mistake failed collection for an
    idle machine.  If ``nvidia-smi`` reports GPUs but not ``index``, the
    observation is unavailable with a configuration-oriented error.
    """
    source = "nvidia-smi"
    scope = f"gpu:index={index}"
    run = runner or _run
    try:
        out = run(_NVIDIA_SMI_CMD)
    except FileNotFoundError:
        return Observation(None, source, error="nvidia-smi not found", scope=scope)
    except subprocess.TimeoutExpired:
        return Observation(None, source, error="nvidia-smi timed out", scope=scope)
    except subprocess.CalledProcessError as exc:
        return Observation(
            None, source, error=f"nvidia-smi exited with status {exc.returncode}",
            scope=scope,
        )
    except OSError as exc:
        return Observation(None, source, error=str(exc), scope=scope)

    gpus: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = _parse_line(line)
        except (ValueError, TypeError):
            parsed = None
        if parsed is None:
            return Observation(
                None, source, error="invalid nvidia-smi output", scope=scope,
            )
        gpus.append(parsed)

    selected = [gpu for gpu in gpus if gpu["index"] == index]
    if gpus and not selected:
        return Observation(
            None, source, error=f"GPU index {index} was not reported", scope=scope,
        )
    return Observation(selected, source, scope=scope)


def gpu_status(
    runner: Optional[Callable[..., str]] = None,
    device_index: int = 0,
) -> list[dict]:
    """Compatibility wrapper returning the selected GPU row or ``[]``.

    Each entry includes ``index``, stable ``uuid``, ``name``, and total/used/free
    MB. Only ``device_index`` is returned, preventing another GPU's free memory
    from satisfying a target for the selected device.

    Returns ``[]`` on any failure — ``nvidia-smi`` absent (FileNotFoundError),
    timeout, non-zero exit, or an unparseable line — which the caller reads as
    "VRAM unknown" while model operations still work.

    ``runner`` is an injectable ``cmd -> stdout`` callable for testing; defaults
    to the real subprocess-backed :func:`_run`.
    """
    observation = observe_gpu(device_index, runner)
    return observation.data or []


def max_free_mb(gpus: list[dict]) -> Optional[int]:
    """Free MB from a selected-GPU snapshot; ``None`` when unavailable.

    The historical name is retained for compatibility. Multiple rows are
    rejected because pooling or maximizing across unrelated devices makes an
    unsafe capacity decision.
    """
    if not gpus:
        return None
    if len(gpus) != 1:
        raise ValueError("expected telemetry for exactly one selected GPU")
    return gpus[0]["free_mb"]
