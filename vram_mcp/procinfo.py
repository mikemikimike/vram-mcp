"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, the ``\\GPU Process Memory(*)`` performance counters (the
source Task Manager uses), summed per PID and joined with ``Get-CimInstance
Win32_Process`` for the name/cmdline, supply both size and label in one call.
``Dedicated Usage`` is real VRAM; ``Non Local Usage`` is VRAM the driver spilled
into system RAM — the severe multi-x slowdown mode, invisible to NVML. Pure
module: every external reader is injected; each degrades to ``[]``/``{}`` on any
failure and never raises.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from ._util import bytes_to_mb, run_capture

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
    """``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline}]`` per GPU-memory
    holder on Windows. ``size_mb`` is Dedicated Usage (real VRAM);
    ``non_local_mb`` is Non Local Usage — VRAM the driver spilled to system RAM,
    the severe multi-x slowdown mode. ``[]`` off-Windows or on any failure. ~1 s:
    all three counters come from ONE sample, so the cost is that of one.
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


def process_table(*, nvml_processes, win_gpu_reader=None,
                  posix_name_reader=None) -> list[dict]:
    """``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline,kind}]`` for GPU VRAM
    holders.

    Starts from NVML (pids + kind + size where available). When
    ``win_gpu_reader`` is given (Windows), its dedicated-VRAM size + spill sizes
    + name + cmdline are authoritative and fill NVML's null sizes and add any
    pids NVML missed. Otherwise ``posix_name_reader`` supplies names for NVML's
    pids and the spill fields stay ``None`` — NVML cannot report them, and
    "unreported" must not read as "no spill".
    """
    table: dict = {}
    for p in nvml_processes():
        table[p["pid"]] = {
            "pid": p["pid"], "size_mb": p.get("size_mb"),
            "shared_mb": None, "non_local_mb": None,
            "name": None, "cmdline": None, "kind": p.get("kind", "compute"),
        }
    if win_gpu_reader is not None:
        for w in win_gpu_reader():
            entry = table.get(w["pid"])
            if entry is None:
                entry = {"pid": w["pid"], "size_mb": None, "shared_mb": None,
                         "non_local_mb": None, "name": None, "cmdline": None,
                         "kind": "compute"}
                table[w["pid"]] = entry
            entry["size_mb"] = w.get("size_mb")
            entry["shared_mb"] = w.get("shared_mb")
            entry["non_local_mb"] = w.get("non_local_mb")
            entry["name"] = w.get("name")
            entry["cmdline"] = w.get("cmdline")
    elif posix_name_reader is not None:
        names = posix_name_reader(list(table.keys()))
        for pid, meta in names.items():
            if pid in table:
                table[pid]["name"] = meta.get("name")
                table[pid]["cmdline"] = meta.get("cmdline")
    return list(table.values())
