# VRAM Audit Log + Meaningful-Model Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give vram-mcp an append-only, cause-attributed audit log ("who removed model X, when") plus size-based meaningful-model detection that resolves per-process VRAM + names on Windows/WDDM (not just Ollama).

**Architecture:** Two new pure modules — `procinfo.py` (sized, named process table: NVML size where present, a Windows perf-counter+CIM fallback where NVML is null) and `audit.py` (one typed append-only `events.jsonl`, a `last_seen.json` diff detector, honest cause attribution). The hardened claims lock/atomic-write helpers are generalized into `_util.py` and shared. `core.py` enriches `other_processes`; `server.py` wires detection into the status tools, logs actions from the mutating tools (with a new `by` owner param), and adds a `history` tool. All new logic is injectable and unit-tested with fakes.

**Tech Stack:** Python 3.10+ (stdlib), `nvidia-ml-py` (already a dep), PowerShell `Get-Counter`/`Get-CimInstance` on Windows via `subprocess`, pytest.

## Global Constraints

- Spec of record: `docs/superpowers/specs/2026-07-16-vram-audit-log-design.md`. Every task implicitly includes it.
- Every new module has **no `mcp` import**, is fully unit-testable with injected fakes (no real GPU/PowerShell/Ollama needed) — matching `gpu.py`/`nvml.py`/`claims.py`.
- **The audit may never break a tool call.** Every resolver/log failure is swallowed; the perf-counter/NVML/file paths degrade to `null`/`[]`/no-op, never raise.
- **Filter is size, not names.** Meaningful holder = process with dedicated VRAM ≥ `VRAM_MCP_MEANINGFUL_MB` (default `512`). Ollama models are always holders regardless of size.
- **NVML-first, perf-counter-fallback for sizes.** The ~1 s `Get-Counter` call runs only inside the status tools' snapshot, only when NVML sizes are null, and only when audit is enabled.
- **One unified typed log** `~/.cache/vram-mcp/events.jsonl`; **`last_seen.json`** for the diff baseline. Bounded at `VRAM_MCP_EVENT_CAP` (default `5000`), pruned on append. Crash-safe atomic writes + the shared file lock.
- **Attribution is honest:** `self_action` (matching recent action), `external` (Ollama, no action), `unattributed` (non-Ollama process). Never invent a culprit.
- **Config (env):** `VRAM_MCP_MEANINGFUL_MB` (512), `VRAM_MCP_EVENT_CAP` (5000), `VRAM_MCP_AUDIT` (on; `0` disables detection + perf-counter).
- Timestamps: UTC ISO 8601 with `Z`. `python -m pytest`, `python -m pyflakes`. Windows Git-Bash env; kill any `vram-mcp.exe` holding the installed script before a reinstall.

## File Structure

- **Modify** `vram_mcp/_util.py` — add generalized `locked`, `iso`, `parse_iso`, `save_json_atomic`, `load_json`, `append_jsonl_capped`, `read_jsonl` (shared by claims + audit).
- **Modify** `vram_mcp/claims.py` — use the `_util` helpers instead of its private copies (behavior identical; its tests are the gate).
- **Create** `vram_mcp/procinfo.py` — `process_table`, `win_gpu_procs`, `posix_names`.
- **Create** `vram_mcp/audit.py` — `meaningful_holders`, `log_action`, `detect_and_log`, `read_events`.
- **Modify** `vram_mcp/core.py` — `combined_status` accepts `procinfo_fn` so `other_processes` carries `size_mb`/`name`/`cmdline`.
- **Modify** `vram_mcp/server.py` — wire procinfo + detection into status tools; `by` param + action logging on mutating tools; `history` tool; env config.
- **Modify** `README.md` — document `history`, the new `other_processes` fields, the env vars.
- **Test:** `tests/test_procinfo.py`, `tests/test_audit.py`, additions to `tests/test_core.py`; `tests/test_claims.py` must stay green through Task 1.

---

### Task 1: Generalize the lock + atomic-write helpers into `_util.py`

**Files:**
- Modify: `vram_mcp/_util.py`
- Modify: `vram_mcp/claims.py`
- Test: `tests/test_util.py` (create), `tests/test_claims.py` (must stay green)

**Interfaces:**
- Produces:
  - `iso(dt: datetime) -> str`; `parse_iso(text: str) -> datetime`
  - `locked(path: Path, timeout: float = 5.0, poll: float = 0.05)` — contextmanager; stale-lock (mtime > `LOCK_STALE_SECONDS`, default 30) breaking; `TimeoutError` on a live lock.
  - `save_json_atomic(path: Path, data) -> None` — temp+`os.replace` with the Windows `PermissionError` retry.
  - `load_json(path: Path, default_factory, is_valid) -> object` — quarantine-on-corrupt (`.corrupt`), returns `default_factory()` when missing/corrupt/invalid.
  - `append_jsonl_capped(path: Path, record: dict, cap: int) -> None` — append one JSON line; if the file exceeds `cap` lines, rewrite to the last `cap`.
  - `read_jsonl(path: Path) -> list[dict]` — parse a `.jsonl` file, skipping unparsable lines; `[]` if missing.

- [ ] **Step 1: Write the failing test** — `tests/test_util.py`

```python
"""Tests for vram_mcp._util shared file/lock/json helpers."""
import json
import os
from datetime import datetime, timezone

import pytest

from vram_mcp import _util


def test_iso_roundtrip():
    dt = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)
    assert _util.iso(dt) == "2026-07-16T18:00:00Z"
    assert _util.parse_iso("2026-07-16T18:00:00Z") == dt


def test_save_and_load_json(tmp_path):
    p = tmp_path / "d.json"
    _util.save_json_atomic(p, {"k": [1, 2]})
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict)) == {"k": [1, 2]}


def test_load_json_missing_returns_default(tmp_path):
    p = tmp_path / "nope.json"
    assert _util.load_json(p, lambda: {"k": []}, lambda d: True) == {"k": []}


def test_load_json_corrupt_quarantines_and_defaults(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("not json {")
    out = _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict))
    assert out == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()  # preserved, not wiped


def test_load_json_wrong_shape_quarantines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([1, 2, 3]))  # valid JSON, wrong shape
    out = _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict) and "k" in d)
    assert out == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()


def test_save_json_retries_replace_on_permission_error(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(_util.os, "replace", flaky)
    _util.save_json_atomic(tmp_path / "d.json", {"ok": True})
    assert calls["n"] == 4


def test_locked_breaks_stale_lock(tmp_path):
    import time
    p = tmp_path / "d.json"
    lock = p.with_suffix(p.suffix + ".lock")
    lock.write_text("9999 old\n")
    old = time.time() - (_util.LOCK_STALE_SECONDS + 30)
    os.utime(lock, (old, old))
    with _util.locked(p, timeout=1.0):
        pass  # acquired by breaking the stale lock
    assert not lock.exists()


def test_locked_times_out_on_live_lock(tmp_path):
    p = tmp_path / "d.json"
    lock = p.with_suffix(p.suffix + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.utime(lock, None)  # fresh -> live
    try:
        with pytest.raises(TimeoutError):
            with _util.locked(p, timeout=0.2, poll=0.05):
                pass
    finally:
        os.close(fd)
        os.remove(lock)


def test_append_jsonl_capped_prunes_to_last_n(tmp_path):
    p = tmp_path / "e.jsonl"
    for i in range(10):
        _util.append_jsonl_capped(p, {"i": i}, cap=5)
    rows = _util.read_jsonl(p)
    assert [r["i"] for r in rows] == [5, 6, 7, 8, 9]


def test_read_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a":1}\nGARBAGE\n{"a":2}\n')
    assert _util.read_jsonl(p) == [{"a": 1}, {"a": 2}]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_util.py -q`
Expected: FAIL — `AttributeError: module 'vram_mcp._util' has no attribute 'iso'`.

- [ ] **Step 3: Implement** — append to `vram_mcp/_util.py`

```python
# --- add to the imports at the top of vram_mcp/_util.py ---
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# --- add below the existing helpers ---

LOCK_STALE_SECONDS = 30.0
_REPLACE_ATTEMPTS = 10
_REPLACE_RETRY_SLEEP = 0.02


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def locked(path: Path, timeout: float = 5.0, poll: float = 0.05):
    """Serialize access to ``path`` via a sibling ``.lock`` file (O_CREAT|O_EXCL,
    cross-platform). A lock older than ``LOCK_STALE_SECONDS`` is presumed
    abandoned (holder hard-killed) and broken. ``TimeoutError`` on a live lock."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock_path).st_mtime
            except OSError:
                continue
            if age > LOCK_STALE_SECONDS:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(poll)
    try:
        os.write(fd, f"{os.getpid()} {iso(_now())}\n".encode("utf-8"))
    except OSError:
        pass
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _quarantine(path: Path) -> None:
    try:
        os.replace(path, path.with_suffix(path.suffix + ".corrupt"))
    except OSError:
        pass


def load_json(path: Path, default_factory, is_valid):
    """Load JSON; on missing/unparsable/invalid-shape return ``default_factory()``.
    A file that EXISTS but is corrupt/wrong-shape is quarantined to ``.corrupt``
    first, so the next write can't silently destroy it."""
    if not path.exists():
        return default_factory()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
        _quarantine(path)
        return default_factory()
    if not is_valid(doc):
        _quarantine(path)
        return default_factory()
    return doc


def save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(_REPLACE_RETRY_SLEEP)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_jsonl_capped(path: Path, record: dict, cap: int) -> None:
    """Append one JSON line; if the file then exceeds ``cap`` lines, rewrite it
    to the last ``cap``. Bounded growth, no daemon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    rows = read_jsonl(path)
    if len(rows) > cap:
        kept = rows[-cap:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return  # best-effort prune; a failed prune never breaks logging
                time.sleep(_REPLACE_RETRY_SLEEP)
```

- [ ] **Step 4: Refactor `claims.py` to use the shared helpers**

In `vram_mcp/claims.py`, delete the private `_iso`, `_parse_iso`, `_locked`, `_quarantine`, `_load`, `_save` (and the `_LOCK_STALE_SECONDS`/`_REPLACE_*` constants) and replace their uses:

```python
# imports
from ._util import (
    locked as _locked, iso as _iso, parse_iso as _parse_iso,
    save_json_atomic, load_json,
)

# _load becomes:
def _load(path: Path) -> dict:
    return load_json(
        path, lambda: {"claims": []},
        lambda d: isinstance(d, dict) and isinstance(d.get("claims"), list),
    )

# _save becomes:
def _save(path: Path, data: dict) -> None:
    save_json_atomic(path, data)
```

Leave `claim`/`renew`/`release`/`list_claims`/`_is_active`/`_prune_expired`/`_default_now` bodies unchanged — they already call `_locked`/`_load`/`_save`/`_iso`/`_parse_iso` by those names.

- [ ] **Step 5: Run tests to verify both pass**

Run: `python -m pytest tests/test_util.py tests/test_claims.py -q`
Expected: PASS (all `test_util` + all existing `test_claims` — the refactor is behavior-preserving). Then `python -m pyflakes vram_mcp/_util.py vram_mcp/claims.py tests/test_util.py`.

- [ ] **Step 6: Commit**

```bash
git add vram_mcp/_util.py vram_mcp/claims.py tests/test_util.py
git commit -m "refactor: generalize lock/atomic-write/jsonl helpers into _util; claims reuses them"
```

---

### Task 2: Sized + named process table (`procinfo.py`)

**Files:**
- Create: `vram_mcp/procinfo.py`
- Test: `tests/test_procinfo.py`

**Interfaces:**
- Produces:
  - `process_table(*, nvml_processes, win_gpu_reader=None, posix_name_reader=None) -> list[dict]` → `[{pid, size_mb, name, cmdline, kind}]`.
  - `win_gpu_procs(timeout: int = 10) -> list[dict]` → `[{pid, size_mb, name, cmdline}]` from the `Dedicated Usage` perf counter joined with `Get-CimInstance` (Windows); `[]` off-Windows / on any failure.
  - `posix_name_reader(pids, timeout: int = 5) -> dict[int, dict]` → `{pid: {"name","cmdline"}}` via `ps` (POSIX).
- Consumes: `nvml.nvml_processes` (`() -> [{pid,size_mb,kind}]`) from Task-independent existing code; `_util.run_capture`, `_util.bytes_to_mb`.

- [ ] **Step 1: Write the failing test** — `tests/test_procinfo.py`

```python
"""Tests for vram_mcp.procinfo — pure, injected readers."""
from vram_mcp import procinfo


def test_process_table_uses_nvml_sizes_when_present():
    # Linux/TCC: NVML gives real sizes; names come from the posix reader.
    nvml = lambda: [{"pid": 100, "size_mb": 8000, "kind": "compute"}]
    names = lambda pids: {100: {"name": "python", "cmdline": "python train.py"}}
    out = procinfo.process_table(nvml_processes=nvml, posix_name_reader=names)
    assert out == [{"pid": 100, "size_mb": 8000, "name": "python",
                    "cmdline": "python train.py", "kind": "compute"}]


def test_process_table_windows_fallback_fills_null_sizes_and_names():
    # WDDM: NVML sizes are null; the win reader supplies size+name+cmdline.
    nvml = lambda: [{"pid": 100, "size_mb": None, "kind": "compute"},
                    {"pid": 200, "size_mb": None, "kind": "graphics"}]
    win = lambda: [{"pid": 100, "size_mb": 14492, "name": "python.exe",
                    "cmdline": "python.exe train_lora_kg.py"}]
    out = procinfo.process_table(nvml_processes=nvml, win_gpu_reader=win)
    by_pid = {p["pid"]: p for p in out}
    assert by_pid[100]["size_mb"] == 14492
    assert by_pid[100]["name"] == "python.exe"
    assert by_pid[100]["kind"] == "compute"          # kind preserved from NVML
    assert by_pid[200]["size_mb"] is None             # win reader didn't see it -> stays null
    assert by_pid[200]["name"] is None


def test_process_table_windows_reader_adds_pid_nvml_missed():
    # The perf counter can see a GPU-memory holder NVML's v3 call didn't list.
    nvml = lambda: []
    win = lambda: [{"pid": 300, "size_mb": 2048, "name": "UnrealEditor.exe",
                    "cmdline": "UnrealEditor.exe Project.uproject"}]
    out = procinfo.process_table(nvml_processes=nvml, win_gpu_reader=win)
    assert out == [{"pid": 300, "size_mb": 2048, "name": "UnrealEditor.exe",
                    "cmdline": "UnrealEditor.exe Project.uproject", "kind": "compute"}]


def test_process_table_no_readers_returns_nvml_only_unnamed():
    nvml = lambda: [{"pid": 100, "size_mb": None, "kind": "compute"}]
    out = procinfo.process_table(nvml_processes=nvml)
    assert out == [{"pid": 100, "size_mb": None, "name": None,
                    "cmdline": None, "kind": "compute"}]


def test_win_gpu_procs_parses_pipe_lines(monkeypatch):
    # pid|bytes|name|cmdline lines from the combined PowerShell reader.
    fake = ("11924|15196000000|python.exe|python.exe train_lora_kg.py\n"
            "1336|1841000000|dwm.exe|dwm.exe\n")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: fake)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    out = {p["pid"]: p for p in procinfo.win_gpu_procs()}
    assert out[11924]["size_mb"] == 14492  # 15.196e9 // 1MB
    assert out[11924]["name"] == "python.exe"
    assert out[1336]["name"] == "dwm.exe"


def test_win_gpu_procs_empty_off_windows(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "linux")
    assert procinfo.win_gpu_procs() == []


def test_win_gpu_procs_empty_on_reader_failure(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: None)
    assert procinfo.win_gpu_procs() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_procinfo.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `vram_mcp/procinfo.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_procinfo.py -q` → PASS. `python -m pyflakes vram_mcp/procinfo.py tests/test_procinfo.py`.

- [ ] **Step 5: Live check (Windows, best-effort)**

Run: `python -c "from vram_mcp.procinfo import process_table; from vram_mcp.nvml import nvml_processes; from vram_mcp.procinfo import win_gpu_procs; import json; print(json.dumps([p for p in process_table(nvml_processes=nvml_processes, win_gpu_reader=win_gpu_procs) if (p['size_mb'] or 0) > 500], indent=2))"`
Expected: prints any real GPU holders ≥ 500 MB with names + sizes (e.g. a `python.exe` training run). If no big holder is resident, an empty list is fine — the point is no exception.

- [ ] **Step 6: Commit**

```bash
git add vram_mcp/procinfo.py tests/test_procinfo.py
git commit -m "feat: procinfo — sized+named GPU process table (NVML + Windows perf-counter fallback)"
```

---

### Task 3: Event log — actions, retention, reads (`audit.py` part 1)

**Files:**
- Create: `vram_mcp/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces:
  - `DEFAULT_EVENTS_PATH`, `DEFAULT_LAST_SEEN_PATH` (`~/.cache/vram-mcp/events.jsonl` / `last_seen.json`).
  - `log_action(*, action, target, kind, actor="unknown", force=False, outcome="ok", detail="", now_fn=..., path=..., cap=5000) -> None` — append one `type="action"` event.
  - `read_events(*, model=None, type=None, limit=50, since=None, path=...) -> list[dict]` — newest-first, filtered.
- Consumes: `_util.iso`, `_util.append_jsonl_capped`, `_util.read_jsonl`, `_util.locked`.

- [ ] **Step 1: Write the failing test** — `tests/test_audit.py`

```python
"""Tests for vram_mcp.audit — event log + diff detector, real tmp_path files."""
from datetime import datetime, timedelta, timezone

from vram_mcp import audit

_T0 = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)


def _clock(start=_T0):
    st = {"now": start}
    def now_fn():
        return st["now"]
    def tick(s):
        st["now"] = st["now"] + timedelta(seconds=s)
    now_fn.tick = tick
    return now_fn


def test_log_action_writes_typed_event(tmp_path):
    p = tmp_path / "events.jsonl"
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", force=True, outcome="ok",
                     detail="unloaded", now_fn=_clock(), path=p)
    events = audit.read_events(path=p)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "action" and e["action"] == "unload"
    assert e["target"] == "qwen3:8b" and e["actor"] == "retro-repo"
    assert e["force"] is True and e["ts"] == "2026-07-16T18:00:00Z"


def test_read_events_newest_first_and_limit(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for name in ["a", "b", "c"]:
        audit.log_action(action="warm", target=name, kind="ollama",
                         now_fn=clk, path=p)
        clk.tick(1)
    got = audit.read_events(limit=2, path=p)
    assert [e["target"] for e in got] == ["c", "b"]


def test_read_events_filter_by_model_and_type(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama", now_fn=clk, path=p)
    audit.log_action(action="warm", target="llama3.2", kind="ollama", now_fn=clk, path=p)
    assert [e["target"] for e in audit.read_events(model="qwen3:8b", path=p)] == ["qwen3:8b"]
    assert [e["action"] for e in audit.read_events(type="action", path=p)]  # all are actions


def test_read_events_since_floor(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="warm", target="old", kind="ollama", now_fn=clk, path=p)
    clk.tick(120)
    audit.log_action(action="warm", target="new", kind="ollama", now_fn=clk, path=p)
    got = audit.read_events(since="2026-07-16T18:01:00Z", path=p)
    assert [e["target"] for e in got] == ["new"]


def test_log_action_respects_cap(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for i in range(8):
        audit.log_action(action="warm", target=str(i), kind="ollama",
                         now_fn=clk, path=p, cap=5)
    targets = [e["target"] for e in audit.read_events(limit=99, path=p)]
    assert targets == ["7", "6", "5", "4", "3"]  # newest-first, last 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audit.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement part 1** — `vram_mcp/audit.py`

```python
"""Append-only, cause-attributed VRAM audit log + disappearance detector.

One typed ``events.jsonl`` (``type`` in action|disappeared|appeared) plus a
``last_seen.json`` diff baseline. Bounded (``cap`` events, pruned on append),
crash-safe (atomic writes + the shared file lock). The audit MUST NEVER break a
tool call: every write is best-effort. Pure w.r.t. time (``now_fn`` injected)
and paths (injected in tests).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ._util import append_jsonl_capped, iso, locked, read_jsonl

DEFAULT_EVENTS_PATH = Path.home() / ".cache" / "vram-mcp" / "events.jsonl"
DEFAULT_LAST_SEEN_PATH = Path.home() / ".cache" / "vram-mcp" / "last_seen.json"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def log_action(*, action: str, target: str, kind: str, actor: str = "unknown",
               force: bool = False, outcome: str = "ok", detail: str = "",
               now_fn=_default_now, path: Path = DEFAULT_EVENTS_PATH,
               cap: int = 5000) -> None:
    """Append one ``type="action"`` event. Best-effort — never raises."""
    event = {
        "ts": iso(now_fn()), "type": "action", "kind": kind, "target": target,
        "action": action, "actor": actor, "force": force, "outcome": outcome,
        "detail": detail,
    }
    try:
        with locked(path):
            append_jsonl_capped(path, event, cap)
    except Exception:
        pass  # the audit may never break a tool call


def read_events(*, model: str | None = None, type: str | None = None,
                limit: int = 50, since: str | None = None,
                path: Path = DEFAULT_EVENTS_PATH) -> list[dict]:
    """Recent events, newest-first, optionally filtered by target/type/since."""
    rows = read_jsonl(path)
    if model is not None:
        rows = [r for r in rows if r.get("target") == model]
    if type is not None:
        rows = [r for r in rows if r.get("type") == type]
    if since is not None:
        rows = [r for r in rows if r.get("ts", "") >= since]  # ISO-Z sorts lexically
    rows.reverse()
    return rows[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audit.py -q` → PASS. `python -m pyflakes vram_mcp/audit.py tests/test_audit.py`.

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/audit.py tests/test_audit.py
git commit -m "feat: audit event log — log_action + read_events, bounded + crash-safe"
```

---

### Task 4: Disappearance detector + attribution (`audit.py` part 2)

**Files:**
- Modify: `vram_mcp/audit.py`
- Modify: `tests/test_audit.py`

**Interfaces:**
- Produces:
  - `meaningful_holders(loaded_models, process_table, threshold_mb) -> list[dict]` → `[{key,kind,target,size_mb}]`; key = `"ollama:<name>"` for every model, `"process:<pid>"` for each process with `size_mb >= threshold_mb`.
  - `detect_and_log(current_holders, *, last_seen_path=..., log_path=..., now_fn=..., cap=5000) -> list[dict]` — diff vs `last_seen.json` under the lock; append `disappeared`/`appeared` events with attribution; rewrite last-seen; return the events emitted.
- Consumes: `_util.save_json_atomic`, `_util.load_json`; its own `read_events` (for recent-action attribution).

- [ ] **Step 1: Write the failing test** — add to `tests/test_audit.py`

```python
def _holder(key, target, kind, size_mb=None):
    return {"key": key, "target": target, "kind": kind, "size_mb": size_mb}


def test_meaningful_holders_all_models_plus_big_processes():
    loaded = [{"name": "qwen3:8b"}]
    procs = [{"pid": 100, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
             {"pid": 200, "size_mb": 40, "name": "chrome.exe", "cmdline": "chrome", "kind": "graphics"}]
    holders = audit.meaningful_holders(loaded, procs, threshold_mb=512)
    keys = {h["key"] for h in holders}
    assert "ollama:qwen3:8b" in keys
    assert "process:100" in keys        # 14.5 GB -> meaningful
    assert "process:200" not in keys    # 40 MB -> noise, filtered


def test_detect_first_run_sets_baseline_no_events(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    holders = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    emitted = audit.detect_and_log(holders, last_seen_path=ls, log_path=log, now_fn=_clock())
    assert emitted == []                       # no false "everything disappeared"
    assert audit.read_events(path=log) == []


def test_detect_disappearance_of_process_is_unattributed(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    p = [_holder("process:100", "python.exe train_lora_kg.py", "process", 14492)]
    audit.detect_and_log(p, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # gone
    assert len(emitted) == 1
    e = emitted[0]
    assert e["type"] == "disappeared" and e["kind"] == "process"
    assert e["cause"] == "unattributed" and e["size_mb"] == 14492
    assert "python.exe train_lora_kg.py" in e["target"]


def test_detect_ollama_disappearance_attributed_to_recent_unload(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(2)
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", now_fn=clk, path=log)             # our action
    clk.tick(1)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "self_action" and emitted[0]["actor"] == "retro-repo"


def test_detect_ollama_disappearance_without_action_is_external(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "external"


def test_detect_appearance_logged(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # empty baseline
    m = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
    emitted = audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["type"] == "appeared" and emitted[0]["target"] == "llama3.2"


def test_detect_second_session_does_not_double_log(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session A logs it
    again = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session B
    assert again == []  # already gone from last_seen -> nothing to re-log
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audit.py -q`
Expected: FAIL — `meaningful_holders`/`detect_and_log` missing.

- [ ] **Step 3: Implement part 2** — add to `vram_mcp/audit.py`

```python
# add imports
from ._util import load_json, parse_iso, save_json_atomic

_RECENT_ACTION_SECONDS = 10.0


def meaningful_holders(loaded_models, process_table, threshold_mb: int) -> list[dict]:
    """Holders worth tracking: every Ollama model, plus non-Ollama processes
    holding >= ``threshold_mb`` dedicated VRAM (size is the filter, not names)."""
    holders = []
    for m in loaded_models:
        name = m.get("name")
        if not name:
            continue
        holders.append({"key": f"ollama:{name}", "target": name,
                        "kind": "ollama", "size_mb": m.get("size_vram_mb")})
    for p in process_table:
        size = p.get("size_mb")
        if size is not None and size >= threshold_mb:
            label = p.get("cmdline") or p.get("name") or f"pid {p['pid']}"
            holders.append({"key": f"process:{p['pid']}", "target": label,
                            "kind": "process", "size_mb": size})
    return holders


def _recent_action_for(target: str, now: datetime, log_path: Path):
    """The most recent unload/ensure_free action on ``target`` within the
    attribution window, else None. ``read_events`` is newest-first."""
    floor = iso(datetime.fromtimestamp(now.timestamp() - _RECENT_ACTION_SECONDS,
                                       tz=timezone.utc))
    actions = [e for e in read_events(model=target, type="action", limit=50, path=log_path)
               if e.get("action") in ("unload", "ensure_free") and e.get("ts", "") >= floor]
    return actions[0] if actions else None


def detect_and_log(current_holders, *, last_seen_path: Path = DEFAULT_LAST_SEEN_PATH,
                   log_path: Path = DEFAULT_EVENTS_PATH, now_fn=_default_now,
                   cap: int = 5000) -> list[dict]:
    """Diff ``current_holders`` against last-seen under the lock; append
    disappeared/appeared events with attribution; rewrite last-seen. Returns the
    events emitted. Best-effort — never raises out to the caller."""
    now = now_fn()
    emitted: list[dict] = []
    try:
        with locked(last_seen_path):
            prev = load_json(last_seen_path, lambda: {"holders": {}},
                             lambda d: isinstance(d, dict) and isinstance(d.get("holders"), dict))
            prev_map = prev["holders"]
            cur_map = {h["key"]: h for h in current_holders}

            for key, h in prev_map.items():
                if key not in cur_map:
                    emitted.append(_disappeared_event(h, now, log_path))
            for key, h in cur_map.items():
                if key not in prev_map:
                    emitted.append({
                        "ts": iso(now), "type": "appeared", "kind": h["kind"],
                        "target": h["target"], "size_mb": h.get("size_mb"),
                        "detail": "now holding VRAM",
                    })

            for e in emitted:
                append_jsonl_capped(log_path, e, cap)
            save_json_atomic(last_seen_path, {"holders": cur_map})
    except Exception:
        return emitted
    return emitted


def _disappeared_event(holder: dict, now: datetime, log_path: Path) -> dict:
    kind = holder["kind"]
    if kind == "ollama":
        action = _recent_action_for(holder["target"], now, log_path)
        if action is not None:
            return {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                    "target": holder["target"], "size_mb": holder.get("size_mb"),
                    "cause": "self_action", "actor": action.get("actor", "unknown"),
                    "detail": f"removed by {action.get('actor','unknown')} via {action.get('action')}"}
        return {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                "target": holder["target"], "size_mb": holder.get("size_mb"),
                "cause": "external",
                "detail": "no vram-mcp action recorded — Ollama idle-expiry, "
                          "memory-pressure eviction, or an external unload."}
    return {"ts": iso(now), "type": "disappeared", "kind": "process",
            "target": holder["target"], "size_mb": holder.get("size_mb"),
            "cause": "unattributed",
            "detail": "process exited or was killed; vram-mcp cannot observe the cause."}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audit.py -q` → PASS (Task 3 + Task 4 tests). `python -m pyflakes vram_mcp/audit.py tests/test_audit.py`.

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/audit.py tests/test_audit.py
git commit -m "feat: audit — meaningful_holders + detect_and_log with honest cause attribution"
```

---

### Task 5: Enrich `other_processes` in `core.py`

**Files:**
- Modify: `vram_mcp/core.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: a `procinfo_fn() -> [{pid,size_mb,name,cmdline,kind}]` (from Task 2, wired in Task 6).
- Produces: `combined_status(..., procinfo_fn=None, ...)` — when given, `other_processes` entries come from `procinfo_fn` (carrying `size_mb`/`name`/`cmdline`/`kind`) instead of the bare NVML list, still excluding resolved Ollama-runner PIDs.

- [ ] **Step 1: Write the failing test** — add to `tests/test_core.py`

```python
def test_combined_status_other_processes_from_procinfo():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4), "expires_at": None}]
    snap = _snap(pid_map={"m1": 555})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda: snap,
        procinfo_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "name": "llama-server.exe", "cmdline": "...", "kind": "compute"},
            {"pid": 999, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
        ],
    )
    # pid 555 is the m1 runner -> excluded; 999 stays, WITH its size + name.
    assert status["other_processes"] == [
        {"pid": 999, "size_mb": 14492, "name": "python.exe",
         "cmdline": "python train.py", "kind": "compute"},
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_core.py::test_combined_status_other_processes_from_procinfo -q`
Expected: FAIL — `combined_status() got an unexpected keyword argument 'procinfo_fn'`.

- [ ] **Step 3: Implement** — in `vram_mcp/core.py`, change `combined_status` to accept `procinfo_fn` and prefer it for `other_processes`:

```python
def combined_status(
    gpu_status_fn, ollama, *,
    snapshot_fn=None, nvml_processes_fn=None, procinfo_fn=None,
) -> dict:
    """... (existing docstring) ... When ``procinfo_fn`` is given, other_processes
    entries carry size_mb/name/cmdline/kind (Task 2's sized+named table); it takes
    precedence over ``nvml_processes_fn``."""
    gpus = gpu_status_fn()
    loaded = _loaded_models(ollama)
    resolved_pids: set = set()
    if snapshot_fn is not None and loaded:
        loaded, resolved_pids = attach_coordination(loaded, snapshot_fn())
    result = {
        "gpus": gpus,
        "loaded": loaded,
        "free_mb": _gpu.max_free_mb(gpus),
    }
    source = procinfo_fn if procinfo_fn is not None else nvml_processes_fn
    if source is not None:
        result["other_processes"] = other_processes(source, resolved_pids)
    return result
```

(`other_processes(fn, exclude_pids)` already filters `fn()` by pid; no change needed there. `nvml_processes_fn` stays supported for callers/tests that don't wire procinfo.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -q` → all PASS (existing + new). `python -m pyflakes vram_mcp/core.py`.

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/core.py tests/test_core.py
git commit -m "feat: core.combined_status accepts procinfo_fn (sized+named other_processes)"
```

---

### Task 6: Server wiring — detection, action logging, `by` param, `history`, config, docs

**Files:**
- Modify: `vram_mcp/server.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `procinfo.process_table`/`win_gpu_procs`/`posix_name_reader`, `nvml.nvml_processes`, `audit.*`, `core.combined_status`/`is_protected`/`ensure_free`.
- Produces (MCP tools): `history(model=None, type=None, limit=50, since=None)`; `unload`/`ensure_free`/`warm` gain optional `by: str = "unknown"`; status tools run detection.

- [ ] **Step 1: Add config + wiring helpers** — near the top of `vram_mcp/server.py`, after `_ollama`:

```python
from . import audit as _audit
from . import procinfo as _procinfo

_AUDIT_ON = os.environ.get("VRAM_MCP_AUDIT", "1") != "0"
_MEANINGFUL_MB = int(os.environ.get("VRAM_MCP_MEANINGFUL_MB", "512"))
_EVENT_CAP = int(os.environ.get("VRAM_MCP_EVENT_CAP", "5000"))


def _procinfo_table() -> list:
    """Sized+named process table; the Windows perf-counter fallback only fires
    when NVML sizes are null (inside win_gpu_procs, which no-ops off-Windows)."""
    return _procinfo.process_table(
        nvml_processes=_nvml.nvml_processes,
        win_gpu_reader=_procinfo.win_gpu_procs,
        posix_name_reader=_procinfo.posix_name_reader,
    )


def _run_detection(status: dict) -> None:
    """Diff the meaningful-holder set from a status snapshot and log changes.
    Best-effort; disabled by VRAM_MCP_AUDIT=0."""
    if not _AUDIT_ON:
        return
    holders = _audit.meaningful_holders(
        status.get("loaded", []), status.get("other_processes", []), _MEANINGFUL_MB)
    _audit.detect_and_log(holders, cap=_EVENT_CAP)
```

- [ ] **Step 2: Wire procinfo + detection into the status tools**

Change `_full_status` to pass `procinfo_fn`, and run detection in `_vram_status_impl` / `_list_loaded_impl`:

```python
def _full_status() -> dict:
    return core.combined_status(
        gpu_status, _ollama,
        snapshot_fn=_snapshot,
        procinfo_fn=(_procinfo_table if _AUDIT_ON else _nvml.nvml_processes),
    )


def _vram_status_impl() -> dict:
    status = _full_status()
    _run_detection(status)
    n_gpu, n_loaded = len(status["gpus"]), len(status["loaded"])
    status["summary"] = (f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
                         f"free: {_fmt_free(status['free_mb'])}.")
    return status


def _list_loaded_impl() -> dict:
    status = core.combined_status(
        lambda: [], _ollama, snapshot_fn=_snapshot,
        procinfo_fn=(_procinfo_table if _AUDIT_ON else None),
    )
    _run_detection(status)
    loaded = status["loaded"]
    return {"loaded": loaded, "summary": f"{len(loaded)} model(s) loaded."}
```

- [ ] **Step 3: Log actions from the mutating tools + add `by`**

In `_unload_impl` (signature `(model, force, by)`), after a real eviction and on the protected-refusal path, log:

```python
def _unload_impl(model: str, force: bool, by: str) -> dict:
    if not force:
        protected, detail = core.is_protected(model, _snapshot())
        if protected:
            _audit.log_action(action="unload", target=model, kind="ollama",
                              actor=by, force=False, outcome="refused",
                              detail="protected (claimed or busy)", cap=_EVENT_CAP)
            summary = (f"'{model}' is protected (claimed or busy); "
                       "pass force=True to override.")
            if detail["busy"] is True and not detail["claims"]:
                summary += (" Note: busy reflects recent GPU activity and can lag a"
                            " few seconds after a generation ends — if the work you"
                            " know about has finished, force=True is safe.")
            return {"ok": False, "model": model, "protected": True, **detail,
                    "summary": summary}
    ok = _ollama.unload(model)
    _audit.log_action(action="unload", target=model, kind="ollama", actor=by,
                      force=force, outcome=("ok" if ok else "failed"),
                      detail=("unloaded" if ok else "ollama unload failed"),
                      cap=_EVENT_CAP)
    return {"ok": ok, "model": model,
            "summary": (f"Unloaded '{model}'." if ok else f"Failed to unload '{model}'.")}
```

Update the `unload` tool to pass `by`:

```python
@mcp.tool()
async def unload(model: str, force: bool = False, by: str = "unknown") -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort busy
    signal — pass ``force=True`` to override (busy is windowed and can lag a few
    seconds past a generation). ``by`` records who requested the eviction in the
    audit log (see ``history``)."""
    return await _in_thread(_unload_impl, model, force, by)
```

In `_ensure_free_impl` (signature `(gb, force, by)`), log one action per model actually unloaded and one per declined:

```python
def _ensure_free_impl(gb: float, force: bool, by: str) -> dict:
    result = core.ensure_free(gb, gpu_status, _ollama, settle=0.5, force=force,
                              snapshot_fn=_snapshot)
    for name in result["unloaded"]:
        _audit.log_action(action="ensure_free", target=name, kind="ollama",
                          actor=by, force=force, outcome="ok",
                          detail=f"unloaded to free {gb} GB", cap=_EVENT_CAP)
    for d in result["declined"]:
        _audit.log_action(action="ensure_free", target=d["name"], kind="ollama",
                          actor=by, force=force, outcome="refused",
                          detail="protected (claimed or busy)", cap=_EVENT_CAP)
    # ... existing summary assembly unchanged ...
    if result["already_free"]:
        base = f"Already {_fmt_free(result['free_mb'])} free (target {gb} GB)."
    elif result["ok"]:
        base = (f"Freed VRAM to {_fmt_free(result['free_mb'])} (target {gb} GB) "
                f"by unloading: {', '.join(result['unloaded']) or 'none'}.")
    else:
        base = (f"Could not reach {gb} GB free (now {_fmt_free(result['free_mb'])}); "
                f"unloaded: {', '.join(result['unloaded']) or 'none'}.")
    declined_note = ""
    if result["declined"]:
        names = ", ".join(d["name"] for d in result["declined"])
        declined_note = f" Protected (force=True to override): {names}."
    result["summary"] = base + declined_note
    return result


@mcp.tool()
async def ensure_free(gb: float, force: bool = False, by: str = "unknown") -> dict:
    """Free VRAM until at least ``gb`` GB is available. Skips claimed/busy models
    unless ``force=True``. ``by`` records the requester in the audit log."""
    return await _in_thread(_ensure_free_impl, gb, force, by)
```

Do the same `by`-param + one `warm` action log for `warm` (`_warm_impl(model, keep_alive, by)`, `outcome="ok"/"failed"`, `action="warm"`).

- [ ] **Step 4: Add the `history` tool**

```python
def _history_impl(model, type_, limit, since) -> dict:
    events = _audit.read_events(model=model, type=type_, limit=limit,
                                since=since, path=_audit.DEFAULT_EVENTS_PATH)
    return {"events": events, "summary": f"{len(events)} event(s)."}


@mcp.tool()
async def history(model: str | None = None, type: str | None = None,
                  limit: int = 50, since: str | None = None) -> dict:
    """The VRAM audit trail, newest first: who ran unload/ensure_free/warm, and
    which models/processes appeared or disappeared (with a best-effort cause).
    Filter by ``model``, ``type`` (action|disappeared|appeared), ``limit``, or an
    ISO ``since`` floor. Answers 'what happened to model X?'."""
    return await _in_thread(_history_impl, model, type, limit, since)
```

- [ ] **Step 5: Smoke test the server end-to-end**

Run:
```bash
python -c "
import asyncio, json
from vram_mcp import server
async def main():
    print((await server.unload('does-not-exist', by='smoke'))['summary'])
    h = await server.history(limit=5)
    print('history:', json.dumps(h['events'][:2]))
    vs = await server.vram_status()
    op = vs.get('other_processes', [])
    print('other_processes sample:', json.dumps(op[:1]))
asyncio.run(main())
"
```
Expected: no exceptions; `history` shows the `unload` action just logged (actor `smoke`); `vram_status` runs (its ~1 s Windows perf-counter cost is expected) and `other_processes` entries carry `name`/`size_mb` keys.

- [ ] **Step 6: Full suite + README**

Run: `python -m pytest -q` → all green. `cd desktop 2>/dev/null; python -m pyflakes vram_mcp/ tests/`.

Update `README.md`: add `history(...)` to the Tools table; note `unload`/`ensure_free`/`warm` gained `by`; add a **Audit trail** subsection under "Multi-session coordination" describing `events.jsonl`, the three causes, and the `VRAM_MCP_MEANINGFUL_MB` / `VRAM_MCP_EVENT_CAP` / `VRAM_MCP_AUDIT` env vars; note `other_processes` now carries real `size_mb`/`name`/`cmdline` (Windows via the perf-counter fallback, ~1 s per status call, disable with `VRAM_MCP_AUDIT=0`).

- [ ] **Step 7: Commit**

```bash
git add vram_mcp/server.py README.md
git commit -m "feat: wire audit — detection on status tools, action logging + by param, history tool, config"
```

---

## Self-Review

**Spec coverage:**
- Size-based meaningful detection, NVML→perf-counter fallback → Task 2 (`procinfo`) + Task 6 (`_procinfo_table` wiring, `VRAM_MCP_MEANINGFUL_MB`). ✓
- Windows dedicated-VRAM via perf counter + names via CIM, one call → Task 2 (`win_gpu_procs` / `_WIN_GPU_PS`). ✓
- `other_processes` gains size/name/cmdline → Task 5 + Task 6. ✓
- Unified typed append-only log, bounded, crash-safe → Task 1 (`append_jsonl_capped`) + Task 3. ✓
- `last_seen` diff detector, one lock, log-once → Task 4 (`detect_and_log`, `test_detect_second_session_does_not_double_log`). ✓
- Honest attribution (self_action/external/unattributed) → Task 4 (`_disappeared_event`, three tests). ✓
- Actions record self-reported actor via `by` → Task 6. ✓
- `history` tool → Task 6. ✓
- Config env vars + opt-out → Task 6 (`_AUDIT_ON`/`_MEANINGFUL_MB`/`_EVENT_CAP`, `_full_status` gating). ✓
- Audit never breaks a tool call → Task 3/4 (`try/except Exception` in `log_action`/`detect_and_log`), Task 6 (`_run_detection` best-effort). ✓
- Generalize claims lock into `_util`, not re-implement → Task 1. ✓
- First-run no-storm → Task 4 (`test_detect_first_run_sets_baseline_no_events`). ✓
- Holder identity keys (`ollama:<name>` / `process:<pid>`) → Task 4 (`meaningful_holders`). ✓
- Perf-counter only on status tools, not fast tools → Task 6 (procinfo wired only into `_full_status`/`_list_loaded_impl`; `unload`/`claim` use `_snapshot` which has no procinfo). ✓

**Placeholder scan:** none — every code step is complete and runnable; no `TBD`/`add error handling` hand-waves.

**Type consistency:** holder dict shape `{key,target,kind,size_mb}` identical in `meaningful_holders` (Task 4), the `_holder` test helper, and `detect_and_log`/`_disappeared_event` consumers. `process_table` entry shape `{pid,size_mb,name,cmdline,kind}` identical across Task 2, Task 5's test, and Task 6's `meaningful_holders` process branch. `log_action` kwargs (`action,target,kind,actor,force,outcome,detail,cap`) identical in Task 3 def and every Task 6 call site. Event `type` values (`action|disappeared|appeared`) and `cause` values (`self_action|external|unattributed`) consistent between Task 4 and the `history` docstring. `by` param threaded as the 3rd positional into `_unload_impl`/`_ensure_free_impl`/`_warm_impl` matching their `_in_thread` calls.

**Open confirmation for the implementer:** verify the exact `Get-Counter` instance-name regex (`pid_(\d+)`) and the `pid|bytes|name|cmdline` line shape against a live run on the target machine before locking `_WIN_GPU_PS` (validated on the dev box during design; PowerShell quoting inside the Python string literal is the fiddly part — test `win_gpu_procs()` live in Task 2 Step 5).
