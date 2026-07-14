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
    """Normalize ``ollama.ps()`` rows to the base per-model status dict.

    ``offloaded_to_cpu`` is True when ``size_vram_mb < total_size_mb`` — part
    of the model spilled to system RAM. When the raw row doesn't carry a
    ``size`` field at all, ``total_size_mb`` is 0 and offload can't be
    detected (reported as False, never a guess of True).
    """
    loaded = []
    for m in ollama.ps():
        size_mb = _bytes_to_mb(m.get("size", 0))
        vram_mb = _bytes_to_mb(m.get("size_vram", 0))
        loaded.append(
            {
                "name": m.get("name"),
                "size_vram_mb": vram_mb,
                "total_size_mb": size_mb,
                "offloaded_to_cpu": vram_mb < size_mb,
                "expires_at": m.get("expires_at"),
            }
        )
    return loaded


def attach_claims(loaded: list[dict], list_claims_fn) -> list[dict]:
    """Attach every active claim (a list, possibly empty) to each model dict."""
    return [{**m, "claims": list_claims_fn(m["name"])} for m in loaded]


def attach_busy(loaded: list[dict], find_pid_fn, busy_fn) -> tuple[list[dict], set]:
    """Attach a best-effort ``busy`` signal to each model dict.

    Returns ``(enriched, resolved_pids)`` — ``resolved_pids`` lets callers
    exclude these PIDs from a general "other processes" survey, since they're
    already represented as Ollama model entries.
    """
    out = []
    resolved: set = set()
    for m in loaded:
        pid = find_pid_fn(m["name"])
        busy = busy_fn(pid) if pid is not None else None
        if pid is not None:
            resolved.add(pid)
        out.append({**m, "busy": busy})
    return out, resolved


def other_processes(nvml_processes_fn, exclude_pids: set) -> list[dict]:
    """Every NVML-visible VRAM holder that isn't an already-listed Ollama model."""
    return [p for p in nvml_processes_fn() if p["pid"] not in exclude_pids]


def combined_status(
    gpu_status_fn: Callable[[], list[dict]], ollama, *,
    list_claims_fn=None, find_pid_fn=None, busy_fn=None, nvml_processes_fn=None,
) -> dict:
    """Snapshot of GPUs + loaded models + best free VRAM.

    Returns ``{"gpus": [...], "loaded": [...], "free_mb": int | None}``, plus
    ``"other_processes"`` when ``nvml_processes_fn`` is given. Each loaded
    model always carries ``total_size_mb``/``offloaded_to_cpu``; it also
    carries ``claims`` when ``list_claims_fn`` is given, and ``busy`` when
    both ``find_pid_fn`` and ``busy_fn`` are given. The optional kwargs let
    ``server.py`` always wire the real implementations in production while
    tests exercise the base case without them.
    """
    gpus = gpu_status_fn()
    loaded = _loaded_models(ollama)
    resolved_pids: set = set()
    if list_claims_fn is not None:
        loaded = attach_claims(loaded, list_claims_fn)
    if find_pid_fn is not None and busy_fn is not None:
        loaded, resolved_pids = attach_busy(loaded, find_pid_fn, busy_fn)
    result = {
        "gpus": gpus,
        "loaded": loaded,
        "free_mb": _gpu.max_free_mb(gpus),
    }
    if nvml_processes_fn is not None:
        result["other_processes"] = other_processes(nvml_processes_fn, resolved_pids)
    return result


def is_protected(
    model_name: str, list_claims_fn, find_pid_fn, busy_fn,
) -> tuple[bool, dict]:
    """Is ``model_name`` unsafe to evict right now?

    Protected if EITHER an active claim exists OR its best-effort ``busy``
    signal is ``True`` — not claim-status alone, so an uncooperative caller
    that never calls ``claim()`` still can't make a real in-flight
    generation trivially interruptible. Returns ``(protected, detail)``,
    ``detail`` = ``{"claims": [...], "busy": bool | None}``.
    """
    active_claims = list_claims_fn(model_name)
    pid = find_pid_fn(model_name)
    busy = busy_fn(pid) if pid is not None else None
    protected = bool(active_claims) or busy is True
    return protected, {"claims": active_claims, "busy": busy}


def ensure_free(
    target_gb: float,
    gpu_status_fn: Callable[[], list[dict]],
    ollama,
    settle: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    *,
    list_claims_fn=None, find_pid_fn=None, busy_fn=None, force: bool = False,
) -> dict:
    """Free VRAM until at least ``target_gb`` is available.

    Fast path: if current max free already meets the target, return without
    unloading anything. Otherwise evict loaded models **largest ``size_vram``
    first**, re-reading free VRAM after each eviction (sleeping ``settle``
    seconds between, if given, to let the driver actually release the memory),
    and stop as soon as the target is met or nothing is left to unload.

    Protection: when ``list_claims_fn``, ``find_pid_fn``, and ``busy_fn`` are
    ALL provided and ``force`` is False, a model with an active claim or a
    ``busy == True`` signal is skipped rather than evicted, and reported in
    ``declined`` (with its claim/busy detail) even if the VRAM target isn't
    fully reached. Protection is a no-op (nothing skipped) if any of the
    three callables is omitted — existing callers see unchanged behavior.

    Returns ``{"ok", "already_free", "free_mb", "unloaded", "declined",
    "target_mb"}``. ``ok`` is ``False`` if the target could not be reached
    (including when VRAM is unknown, i.e. ``free_mb is None``, so we cannot
    prove success).

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
            "declined": [],
            "target_mb": target_mb,
        }

    # Largest-first so we free the most VRAM with the fewest evictions.
    models = sorted(
        _loaded_models(ollama),
        key=lambda m: m["size_vram_mb"],
        reverse=True,
    )

    protection_enabled = (
        not force and list_claims_fn is not None
        and find_pid_fn is not None and busy_fn is not None
    )

    unloaded: list[str] = []
    declined: list[dict] = []
    for m in models:
        name = m["name"]
        if not name:
            continue
        if protection_enabled:
            protected, detail = is_protected(name, list_claims_fn, find_pid_fn, busy_fn)
            if protected:
                declined.append({"name": name, **detail})
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
        "declined": declined,
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
