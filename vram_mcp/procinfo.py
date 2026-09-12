"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, the ``\\GPU Process Memory(*)`` performance counters (the
source Task Manager uses), summed per PID and joined with ``Get-CimInstance
Win32_Process``, can label holders. Their memory totals aggregate adapters, so
they are never assigned to a selected GPU without an adapter mapping. Pure
module: every external reader is injected; each degrades to ``[]``/``{}`` on any
failure and never raises.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from ._util import bytes_to_mb, run_capture
from .observations import Observation
from . import nvml as _nvml

# ONE Get-Counter call sampling all three counters, so the ~1 s perf-counter
# cost is paid once. Samples are discriminated by their Path (which the counter
# subsystem lowercases; -match is case-insensitive anyway).
_WIN_GPU_PS = (
    "$paths=@('\\GPU Process Memory(*)\\Dedicated Usage',"
    "'\\GPU Process Memory(*)\\Shared Usage',"
    "'\\GPU Process Memory(*)\\Non Local Usage');"
    "$d=@{};$s=@{};$n=@{};"
    "(Get-Counter -Counter $paths -EA SilentlyContinue).CounterSamples | "
    "Where-Object { $_.CookedValue -gt 0 -and $_.InstanceName -match 'pid_(\\d+)' } | "
    "ForEach-Object { "
    "$id=[int]($_.InstanceName -replace '.*pid_(\\d+).*','$1'); "
    "$v=[int64]$_.CookedValue; "
    "if($_.Path -match 'non local usage'){ $n[$id]=[int64]$n[$id]+$v } "
    "elseif($_.Path -match 'shared usage'){ $s[$id]=[int64]$s[$id]+$v } "
    "else { $d[$id]=[int64]$d[$id]+$v } };"
    "$ids=@($d.Keys)+@($s.Keys)+@($n.Keys) | Sort-Object -Unique;"
    "foreach($id in $ids){ $p=Get-CimInstance Win32_Process -Filter "
    "\"ProcessId=$id\" -EA SilentlyContinue; "
    "'{0}|{1}|{2}|{3}|{4}|{5}' -f "
    "$id,[int64]$d[$id],[int64]$s[$id],[int64]$n[$id],$p.Name,$p.CommandLine }"
)


def _run_powershell(command: str, timeout: int) -> Optional[str]:
    return run_capture(["powershell", "-NoProfile", "-Command", command], timeout)


def win_gpu_procs(timeout: int = 10) -> list[dict]:
    """Return adapter-aggregated Windows GPU counters and process identity.

    ``size_mb`` is Dedicated Usage and ``non_local_mb`` is Non Local Usage, but
    neither identifies an adapter. Selected-device callers may use `name` and
    `cmdline`; they must leave these memory fields unavailable. Returns ``[]``
    off-Windows or on any failure. All counters come from one ~1-second sample.
    """
    if sys.platform != "win32":
        return []
    out_text = _run_powershell(_WIN_GPU_PS, timeout)
    if not out_text:
        return []
    procs = []
    for line in out_text.splitlines():
        # cmdline is last and unsplit: a command line may itself contain '|'.
        parts = line.strip().split("|", 5)
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        pid, dedicated, shared, non_local, name, cmdline = parts
        procs.append({
            "pid": int(pid),
            "size_mb": bytes_to_mb(dedicated, default=None),
            "shared_mb": bytes_to_mb(shared, default=None),
            "non_local_mb": bytes_to_mb(non_local, default=None),
            "name": name.strip() or None,
            "cmdline": cmdline.strip() or None,
        })
    return procs


_PS_NAME = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*)$")


def posix_name_reader(pids, timeout: int = 5) -> dict:
    """``{pid: {"name","cmdline"}}`` via ``ps`` for the given pids (POSIX)."""
    wanted = set(pids)
    if not wanted:
        return {}
    stdout = run_capture(["ps", "-eo", "pid,comm,args"], timeout)
    if stdout is None:
        return {}
    names = {}
    for line in stdout.splitlines()[1:]:
        m = _PS_NAME.match(line)
        if not m:
            continue
        pid = int(m.group(1))
        if pid in wanted:
            names[pid] = {"name": m.group(2), "cmdline": m.group(3).strip()}
    return names


def _base_table(rows: list[dict]) -> dict[int, dict]:
    """Normalize selected-device NVML rows into the public process shape."""
    return {
        p["pid"]: {
            "pid": p["pid"], "size_mb": p.get("size_mb"),
            "shared_mb": None, "non_local_mb": None,
            "name": None, "cmdline": None, "kind": p.get("kind", "compute"),
        }
        for p in rows
    }


def _enrich_processes(
    rows: list[dict], *, platform: str, win_gpu_reader=None,
    posix_reader=None,
) -> list[dict]:
    """Attach process identity without weakening selected-GPU attribution.

    Windows GPU Process Memory counters aggregate a PID across adapters. Their
    name and command line are useful, but their memory values cannot be assigned
    to one selected GPU. Consequently they never fill memory fields or add a
    PID that NVML did not report for the selected device.
    """
    table = _base_table(rows)
    if platform == "win32" and win_gpu_reader is not None:
        for windows_row in win_gpu_reader():
            entry = table.get(windows_row["pid"])
            if entry is None:
                continue
            entry["name"] = windows_row.get("name")
            entry["cmdline"] = windows_row.get("cmdline")
    elif platform != "win32" and posix_reader is not None:
        names = posix_reader(list(table.keys()))
        for pid, meta in names.items():
            if pid in table:
                table[pid]["name"] = meta.get("name")
                table[pid]["cmdline"] = meta.get("cmdline")
    return list(table.values())


def observe_processes(
    index: int = 0,
    *,
    nvml=None,
    platform: Optional[str] = None,
    nvml_observer=None,
    win_gpu_reader=None,
    posix_reader=None,
) -> Observation[list[dict]]:
    """Observe process holders for one selected GPU with platform enrichment."""
    platform = sys.platform if platform is None else platform
    nvml_observer = nvml_observer or _nvml.observe_processes
    win_gpu_reader = win_gpu_procs if win_gpu_reader is None else win_gpu_reader
    posix_reader = posix_name_reader if posix_reader is None else posix_reader
    nvml_observation = nvml_observer(index, nvml=nvml)
    source = "nvml+windows-process-counters" if platform == "win32" else "nvml+ps"
    scope = f"gpu:index={index}"
    coverage = {**(nvml_observation.coverage or {}), "non_local_memory": False}
    if not nvml_observation.known:
        return Observation(
            None, source, observed_at=nvml_observation.observed_at,
            error=nvml_observation.error or "NVML process telemetry unavailable",
            scope=scope, coverage=coverage,
        )
    rows = _enrich_processes(
        nvml_observation.data or [], platform=platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_reader,
    )
    return Observation(
        rows, source, observed_at=nvml_observation.observed_at, scope=scope,
        coverage=coverage,
    )


def process_table(*, nvml_processes, win_gpu_reader=None,
                  posix_name_reader=None, platform: Optional[str] = None) -> list[dict]:
    """``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline,kind}]`` for GPU VRAM
    holders.

    Starts from selected-device NVML rows. Platform dispatch is explicit:
    Windows counters supply names only, while POSIX uses ``ps``. Memory fields
    from adapter-aggregated Windows counters remain ``None`` unless a future
    collector can prove adapter attribution.
    """
    selected_platform = sys.platform if platform is None else platform
    return _enrich_processes(
        nvml_processes(), platform=selected_platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_name_reader,
    )
