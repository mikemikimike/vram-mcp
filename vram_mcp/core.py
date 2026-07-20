"""Pure orchestration over a GPU-status function and an Ollama client.

Nothing here touches the network or a real GPU directly: callers inject a
``gpu_status_fn`` (``() -> list[dict]``) and an ``ollama`` client, so the whole
module is exercisable with plain fakes in tests. No ``mcp`` import.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from . import gpu as _gpu
from ._util import bytes_to_mb as _shared_bytes_to_mb

_MB_PER_GB = 1024

SPILL_THRESHOLD_MB = 256   # below this, non-local usage is normal desktop noise
TIGHT_MB = 1024


def _bytes_to_mb(value) -> int:
    """Bytes -> whole MB; 0 for missing/garbage (a number is always expected here)."""
    return _shared_bytes_to_mb(value, default=0)


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


class Snapshot:
    """One consistent capture of the coordination signals, taken ONCE per
    operation and shared across every model it touches (the data cannot
    meaningfully change within a single call, so per-model re-collection
    would only add subprocess/IO churn). All three inputs are plain data,
    so a Snapshot is trivially fake-able in tests.

    * ``all_claims`` — every active claim record (one ledger read).
    * ``pid_map`` — Ollama tag → runner PID (one process listing + one
      manifest walk; aliases sharing a blob all map to the runner's PID).
    * ``busy_map`` — runner PID → ``True | False | None`` (one NVML session).
    """

    def __init__(self, all_claims: list[dict], pid_map: dict,
                 busy_map: dict) -> None:
        self.all_claims = all_claims
        self.pid_map = pid_map
        self.busy_map = busy_map

    @classmethod
    def capture(cls, all_claims_fn, pid_map_fn, busy_map_fn) -> "Snapshot":
        """Run the three collectors once. ``busy_map_fn`` receives the PIDs
        the pid_map surfaced, so the NVML fetch covers exactly what's needed."""
        all_claims = all_claims_fn() if all_claims_fn else []
        pid_map = pid_map_fn() if pid_map_fn else {}
        pids = sorted(set(pid_map.values()))
        busy_map = busy_map_fn(pids) if (busy_map_fn and pids) else {}
        return cls(all_claims, pid_map, busy_map)

    def claims_for(self, model_name) -> list[dict]:
        """Active claims on ``model_name``; [] for a None/unknown name (a
        nameless ps() row must never be attributed everyone's claims)."""
        if not model_name:
            return []
        return [c for c in self.all_claims
                if isinstance(c, dict) and c.get("model") == model_name]

    def pid_for(self, model_name):
        if not model_name:
            return None
        return self.pid_map.get(model_name)

    def busy_for(self, model_name):
        pid = self.pid_for(model_name)
        if pid is None:
            return None
        return self.busy_map.get(pid)


def attach_coordination(loaded: list[dict], snap: Snapshot) -> tuple[list[dict], set]:
    """Attach ``claims`` + ``busy`` to each model dict from one Snapshot.

    Returns ``(enriched, resolved_pids)`` — ``resolved_pids`` lets callers
    exclude Ollama-runner PIDs from a general "other processes" survey, since
    they're already represented as model entries.
    """
    out = []
    resolved: set = set()
    for m in loaded:
        name = m.get("name")
        pid = snap.pid_for(name)
        if pid is not None:
            resolved.add(pid)
        out.append({**m, "claims": snap.claims_for(name), "busy": snap.busy_for(name)})
    return out, resolved


def other_processes(nvml_processes_fn, exclude_pids: set) -> list[dict]:
    """Every NVML-visible VRAM holder that isn't an already-listed Ollama model."""
    return [p for p in nvml_processes_fn() if p["pid"] not in exclude_pids]


def pressure(gpus: list[dict], loaded: list[dict], other_processes: list[dict],
             *, spill_threshold_mb: int = SPILL_THRESHOLD_MB,
             tight_mb: int = TIGHT_MB) -> dict:
    """Classify how close the GPU is to a slow-mode, and which one.

    Two independent degradations exist and must not be conflated:

    * **Driver-forced spill** — WDDM demand-pages VRAM to system RAM under
      pressure (``non_local_mb``). This is the severe multi-x slowdown.
    * **Deliberate CPU offload** — Ollama chose to put layers on the CPU
      backend up front (``offloaded_to_cpu``). Slower, but a considered
      placement, not paging.

    Returns ``{"state", "free_mb", "non_local_mb", "spilling",
    "offloaded_models", "detail"}``. ``state`` is the first match of
    thrashing > degraded > tight > ok.
    """
    free_mb = _gpu.max_free_mb(gpus)
    non_local_mb = sum(
        p.get("non_local_mb") or 0
        for p in other_processes if isinstance(p, dict)
    )
    spilling = non_local_mb >= spill_threshold_mb
    offloaded = [
        m["name"] for m in loaded
        if isinstance(m, dict) and m.get("offloaded_to_cpu") and m.get("name")
    ]

    if spilling:
        state = "thrashing"
        detail = (
            f"{non_local_mb} MB of VRAM has spilled to system RAM; the driver "
            "is paging. Expect severe slowdown — free VRAM or reduce load."
        )
    elif offloaded:
        state = "degraded"
        detail = (
            f"Model(s) partly on CPU: {', '.join(offloaded)}. Slower than "
            "full GPU residency, but a deliberate Ollama placement, not paging."
        )
    elif free_mb is not None and free_mb < tight_mb:
        state = "tight"
        detail = (
            f"Only {free_mb} MB free; the next load will likely spill or fail."
        )
    else:
        state = "ok"
        detail = "No VRAM pressure detected."

    return {
        "state": state,
        "free_mb": free_mb,
        "non_local_mb": non_local_mb,
        "spilling": spilling,
        "offloaded_models": offloaded,
        "detail": detail,
    }


def combined_status(
    gpu_status_fn: Callable[[], list[dict]], ollama, *,
    snapshot_fn=None, nvml_processes_fn=None, procinfo_fn=None,
) -> dict:
    """Snapshot of GPUs + loaded models + best free VRAM.

    Returns ``{"gpus": [...], "loaded": [...], "free_mb": int | None,
    "pressure": {...}}``, plus ``"other_processes"`` when ``nvml_processes_fn``
    or ``procinfo_fn`` is given. ``pressure`` is ALWAYS present: it is computed
    from the process table when one is available (driver-spill detection needs
    ``non_local_mb``, which only that table carries) and from the GPU + model
    data alone otherwise — in which case it can still report ``degraded``/
    ``tight``/``ok``, just never ``thrashing``.
    Each loaded model always carries ``total_size_mb``/``offloaded_to_cpu``; when
    ``snapshot_fn`` (``() -> Snapshot``) is given it also carries ``claims``
    and ``busy``, all derived from ONE snapshot capture rather than per-model
    re-collection. When ``procinfo_fn`` is given, other_processes entries carry
    size_mb/name/cmdline/kind (Task 2's sized+named table); it takes precedence
    over ``nvml_processes_fn``. Omitting the kwargs gives the plain base shape
    (tests, ``advise``).
    """
    gpus = gpu_status_fn()
    loaded = _loaded_models(ollama)
    resolved_pids: set = set()
    # Nothing loaded -> nothing to enrich; skip the snapshot's subprocess/IO.
    if snapshot_fn is not None and loaded:
        loaded, resolved_pids = attach_coordination(loaded, snapshot_fn())
    result = {
        "gpus": gpus,
        "loaded": loaded,
        "free_mb": _gpu.max_free_mb(gpus),
    }
    source = procinfo_fn if procinfo_fn is not None else nvml_processes_fn
    procs: list[dict] = []
    if source is not None:
        procs = other_processes(source, resolved_pids)
        result["other_processes"] = procs
    result["pressure"] = pressure(gpus, loaded, procs)
    return result


def is_protected(model_name: str, snap: Snapshot) -> tuple[bool, dict]:
    """Is ``model_name`` unsafe to evict right now?

    Protected if EITHER an active claim exists OR its best-effort ``busy``
    signal is ``True`` — not claim-status alone, so an uncooperative caller
    that never calls ``claim()`` still can't make a real in-flight
    generation trivially interruptible. Returns ``(protected, detail)``,
    ``detail`` = ``{"claims": [...], "busy": bool | None}``.
    """
    active_claims = snap.claims_for(model_name)
    busy = snap.busy_for(model_name)
    protected = bool(active_claims) or busy is True
    return protected, {"claims": active_claims, "busy": busy}


def reserved_mb(all_claims: list[dict]) -> int:
    """Total VRAM (MB) spoken for by active reservations.

    ``all_claims`` is the ledger's already-expiry-filtered list, so every
    reservation here is live. Malformed records are skipped rather than
    raising — one unusable record must not break a status call.
    """
    total = 0.0
    for record in all_claims:
        if not isinstance(record, dict) or record.get("kind") != "reservation":
            continue
        try:
            total += float(record["gb"])
        except (KeyError, TypeError, ValueError):
            continue
    return int(round(total * _MB_PER_GB))


def ensure_free(
    target_gb: float,
    gpu_status_fn: Callable[[], list[dict]],
    ollama,
    settle: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    *,
    snapshot_fn=None, force: bool = False,
) -> dict:
    """Free VRAM until at least ``target_gb`` is available.

    Fast path: if current max free already meets the target, return without
    unloading anything. Otherwise evict loaded models **largest ``size_vram``
    first**, re-reading free VRAM after each eviction (sleeping ``settle``
    seconds after each SUCCESSFUL unload, if given, to let the driver
    actually release the memory), and stop as soon as the target is met or
    nothing is left to unload.

    Protection: when ``snapshot_fn`` (``() -> Snapshot``) is provided and
    ``force`` is False, the coordination snapshot is captured ONCE before the
    loop, and any model with an active claim or ``busy == True`` is skipped
    rather than evicted, reported in ``declined`` (with claim/busy detail)
    even if the VRAM target isn't fully reached. Protection is a no-op when
    ``snapshot_fn`` is omitted.

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

    # ONE snapshot for the whole eviction pass (protection data cannot
    # meaningfully change mid-call), and none at all when there are no
    # candidate models to protect.
    snap = (snapshot_fn()
            if (snapshot_fn is not None and not force and models) else None)

    unloaded: list[str] = []
    declined: list[dict] = []
    for m in models:
        name = m["name"]
        if not name:
            continue
        if snap is not None:
            protected, detail = is_protected(name, snap)
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
