"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, the ``\\GPU Process Memory(*)\\Dedicated Usage`` performance
counter (the source Task Manager uses), summed per PID and joined with
``Get-CimInstance Win32_Process`` for the name/cmdline, supplies both size and
label in one call. Pure module: every external reader is injected; each degrades
to ``[]``/``{}`` on any failure and never raises.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from ._util import bytes_to_mb, run_capture

# One PowerShell call: sum Dedicated Usage per pid, join name+cmdline.
_WIN_GPU_PS = (
    "$m=@{};"
    "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -EA SilentlyContinue)"
    ".CounterSamples | Where-Object { $_.CookedValue -gt 0 -and "
    "$_.InstanceName -match 'pid_(\\d+)' } | ForEach-Object { "
    "$id=[int]($_.InstanceName -replace '.*pid_(\\d+).*','$1'); "
    "$m[$id]=[int64]$m[$id]+[int64]$_.CookedValue };"
    "foreach($id in $m.Keys){ $p=Get-CimInstance Win32_Process -Filter "
    "\"ProcessId=$id\" -EA SilentlyContinue; "
    "'{0}|{1}|{2}|{3}' -f $id,$m[$id],$p.Name,$p.CommandLine }"
)


def _run_powershell(command: str, timeout: int) -> Optional[str]:
    return run_capture(["powershell", "-NoProfile", "-Command", command], timeout)


def win_gpu_procs(timeout: int = 10) -> list[dict]:
    """``[{pid,size_mb,name,cmdline}]`` for every dedicated-VRAM holder on Windows.
    ``[]`` off-Windows or on any failure. ~1 s (a full perf-counter sample)."""
    if sys.platform != "win32":
        return []
    out_text = _run_powershell(_WIN_GPU_PS, timeout)
    if not out_text:
        return []
    procs = []
    for line in out_text.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        pid, raw_bytes, name, cmdline = parts
        procs.append({
            "pid": int(pid),
            "size_mb": bytes_to_mb(raw_bytes, default=None),
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
    """``[{pid,size_mb,name,cmdline,kind}]`` for GPU VRAM holders.

    Starts from NVML (pids + kind + size where available). When
    ``win_gpu_reader`` is given (Windows), its dedicated-VRAM size + name +
    cmdline are authoritative and fill NVML's null sizes and add any pids NVML
    missed. Otherwise ``posix_name_reader`` supplies names for NVML's pids.
    """
    table: dict = {}
    for p in nvml_processes():
        table[p["pid"]] = {
            "pid": p["pid"], "size_mb": p.get("size_mb"),
            "name": None, "cmdline": None, "kind": p.get("kind", "compute"),
        }
    if win_gpu_reader is not None:
        for w in win_gpu_reader():
            entry = table.get(w["pid"])
            if entry is None:
                entry = {"pid": w["pid"], "size_mb": None, "name": None,
                         "cmdline": None, "kind": "compute"}
                table[w["pid"]] = entry
            entry["size_mb"] = w.get("size_mb")
            entry["name"] = w.get("name")
            entry["cmdline"] = w.get("cmdline")
    elif posix_name_reader is not None:
        names = posix_name_reader(list(table.keys()))
        for pid, meta in names.items():
            if pid in table:
                table[pid]["name"] = meta.get("name")
                table[pid]["cmdline"] = meta.get("cmdline")
    return list(table.values())
