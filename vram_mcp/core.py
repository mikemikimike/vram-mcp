"""Pure orchestration over a GPU-status function and an Ollama client.

Nothing here touches the network or a real GPU directly: callers inject a
``gpu_status_fn`` (``() -> list[dict]``) and an ``ollama`` client, so the whole
module is exercisable with plain fakes in tests. No ``mcp`` import.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from . import gpu as _gpu

_BYTES_PER_MB = 1024 * 1024
_MB_PER_GB = 1024


def _bytes_to_mb(value) -> int:
    """Best-effort bytes -> whole MB, tolerating missing/garbage values."""
    try:
        return int(value) // _BYTES_PER_MB
    except (TypeError, ValueError):
        return 0


def _loaded_models(ollama) -> list[dict]:
    """Normalize ``ollama.ps()`` rows to ``{name, size_vram_mb, expires_at}``."""
    loaded = []
    for m in ollama.ps():
        loaded.append(
            {
                "name": m.get("name"),
                "size_vram_mb": _bytes_to_mb(m.get("size_vram", 0)),
                "expires_at": m.get("expires_at"),
            }
        )
    return loaded


def combined_status(gpu_status_fn: Callable[[], list[dict]], ollama) -> dict:
    """Snapshot of GPUs + loaded models + best free VRAM.

    Returns ``{"gpus": [...], "loaded": [...], "free_mb": int | None}`` where
    ``free_mb`` is the max free across GPUs, or ``None`` when VRAM is unknown.
    """
    gpus = gpu_status_fn()
    return {
        "gpus": gpus,
        "loaded": _loaded_models(ollama),
        "free_mb": _gpu.max_free_mb(gpus),
    }


def ensure_free(
    target_gb: float,
    gpu_status_fn: Callable[[], list[dict]],
    ollama,
    settle: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Free VRAM until at least ``target_gb`` is available.

    Fast path: if current max free already meets the target, return without
    unloading anything. Otherwise evict loaded models **largest ``size_vram``
    first**, re-reading free VRAM after each eviction (sleeping ``settle``
    seconds between, if given, to let the driver actually release the memory),
    and stop as soon as the target is met or nothing is left to unload.

    Returns ``{"ok", "already_free", "free_mb", "unloaded", "target_mb"}``.
    ``ok`` is ``False`` if the target could not be reached (including when VRAM
    is unknown, i.e. ``free_mb is None``, so we cannot prove success).

    ``sleep`` is injected so tests never actually wait.
    """
    target_mb = int(round(target_gb * _MB_PER_GB))

    def current_free() -> Optional[int]:
        return _gpu.max_free_mb(gpu_status_fn())

    free = current_free()
    if free is not None and free >= target_mb:
        return {
            "ok": True,
            "already_free": True,
            "free_mb": free,
            "unloaded": [],
            "target_mb": target_mb,
        }

    # Largest-first so we free the most VRAM with the fewest evictions.
    models = sorted(
        _loaded_models(ollama),
        key=lambda m: m["size_vram_mb"],
        reverse=True,
    )

    unloaded: list[str] = []
    for m in models:
        name = m["name"]
        if not name:
            continue
        if ollama.unload(name):
            unloaded.append(name)
        if settle:
            sleep(settle)
        free = current_free()
        if free is not None and free >= target_mb:
            break

    ok = free is not None and free >= target_mb
    return {
        "ok": ok,
        "already_free": False,
        "free_mb": free,
        "unloaded": unloaded,
        "target_mb": target_mb,
    }


def _expires_is_forever(expires_at) -> bool:
    """True if ``expires_at`` denotes an effectively-never expiry (pinned).

    Ollama uses a far-future / zero-year timestamp for ``keep_alive=-1``.
    """
    if expires_at in (None, "", "forever"):
        return False
    text = str(expires_at)
    # Ollama emits e.g. "0001-01-01T00:00:00Z" for a never-expiring model, and
    # far-future years for very long keep-alives.
    if text.startswith("0001-01-01"):
        return True
    year = text[:4]
    if year.isdigit() and int(year) >= 9999:
        return True
    return False


def advise(gpu_status_fn: Callable[[], list[dict]], ollama) -> dict:
    """Heuristic suggestions for keeping VRAM healthy.

    Returns ``{"suggestions": [str, ...]}``. Empty list means nothing to flag.
    """
    status = combined_status(gpu_status_fn, ollama)
    loaded = status["loaded"]
    free_mb = status["free_mb"]

    suggestions: list[str] = []

    # Multiple models resident while free VRAM is low (or unknown).
    low_free = free_mb is None or free_mb < 2 * _MB_PER_GB
    if len(loaded) > 1 and low_free:
        suggestions.append(
            "Multiple models are loaded and free VRAM is low; set "
            "OLLAMA_MAX_LOADED_MODELS=1 to keep only one model resident."
        )

    # Any model pinned effectively forever.
    pinned = [m["name"] for m in loaded if _expires_is_forever(m["expires_at"])]
    if pinned:
        names = ", ".join(str(n) for n in pinned)
        suggestions.append(
            f"Model(s) pinned in VRAM indefinitely ({names}); set a finite "
            "OLLAMA_KEEP_ALIVE (e.g. 5m) so idle models release VRAM."
        )

    return {"suggestions": suggestions}
