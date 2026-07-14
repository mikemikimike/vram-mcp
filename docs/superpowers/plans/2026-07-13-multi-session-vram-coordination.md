# Multi-Session VRAM Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let independent `vram-mcp` sessions coordinate one GPU's VRAM by adding a self-reported claim ledger (who's using a model, why), NVML-based best-effort busy detection, and full VRAM-consumer visibility (not just Ollama models).

**Architecture:** Three new pure modules (`nvml.py`, `ollama_correlate.py`, `claims.py`) mirror the existing `gpu.py`/`ollama.py` pattern — injectable, no `mcp` import, degrade to `None`/`[]` on any failure. `core.py` stays the sole orchestrator, combining all four data sources into the per-model dicts and the eviction-protection decision. `server.py` wires the real implementations and exposes 4 new tools.

**Tech Stack:** `nvidia-ml-py` (new dependency, NVIDIA-official NVML bindings), stdlib `subprocess`/`json`/`pathlib`/`uuid`/`datetime` — no other new dependencies.

## Global Constraints

- New dependency `nvidia-ml-py` — confirmed acceptable; add to `pyproject.toml` `dependencies`.
- Every new pure module (`nvml.py`, `ollama_correlate.py`) has **no `mcp` import**, is fully unit-testable with injected fakes — no real GPU, driver, or Ollama daemon required to run the suite (matches `gpu.py`/`ollama.py`'s existing pattern exactly).
- Attribution is **self-reported only** (a `claim()` tool call) — never inferred. Busy detection is **NVML-based and windowed** (`nvmlDeviceGetProcessUtilization(handle, 0)` reads NVML's own short sample buffer, not a single instant) — never requires wrapping a caller's actual inference calls.
- A model is protected from default eviction if **an active claim exists OR its best-effort `busy` signal is `True`** — not claim-status alone. `force=True` bypasses both, on both `unload()` and `ensure_free()`.
- Claims expire via **TTL, not explicit release only** — a crashed session's claim simply stops being active once `expires_at` passes; no cleanup step required.
- PID→model-name correlation (`ollama_correlate.py`) is **isolated in its own module because it's undocumented/version-dependent** (verified against Ollama 0.31.1 on this machine) — every function degrades to `None` on any failure rather than guessing, so a future Ollama layout change only turns `busy` into "unknown," never breaks anything else.
- Every timestamp is **UTC ISO 8601 with a `Z` suffix**, matching Ollama's own `expires_at` format.
- `owner` (on a claim) is a **free-form human-readable label**, not a machine-parsed session ID.
- On Windows/WDDM, NVML's per-process VRAM size (`usedGpuMemory`) is `None` — **verified empirically on the dev machine**, this is not a hypothetical edge case. `other_processes[].size_mb` must be `int | None` from the start, never assumed present.
- **Naming:** the per-model claims field is `claims` (a list, possibly empty) — plural, matching `core.is_protected`'s `detail["claims"]` — never a singular `claim`. This refines the spec's wording (which said singular "claim") for consistency across the codebase; see Self-Review.
- Out of scope (per spec): NVML accounting mode; any change to how calling sessions issue real Ollama inference requests; non-NVIDIA GPU backends.

---

## File Structure

- **Create** `vram_mcp/nvml.py` — NVML wrapper: accurate memory info, all VRAM-holding processes, windowed busy signal.
- **Create** `vram_mcp/ollama_correlate.py` — PID→Ollama-model-tag correlation.
- **Create** `vram_mcp/claims.py` — shared, file-based, TTL-based attribution ledger.
- **Modify** `vram_mcp/core.py` — offload detection, claim/busy/other-process enrichment, protection-aware `ensure_free`.
- **Modify** `vram_mcp/server.py` — wire the new modules; add `claim`/`renew`/`release`/`list_claims` tools; update `vram_status`/`list_loaded`/`unload`/`ensure_free`.
- **Modify** `pyproject.toml` — add `nvidia-ml-py` dependency.
- **Modify** `README.md` — document the 4 new tools and the new `force` parameter.
- **Test**: `tests/test_nvml.py`, `tests/test_ollama_correlate.py`, `tests/test_claims.py`, plus additions to `tests/test_core.py`.

---

### Task 1: NVML wrapper (`nvml.py`)

**Files:**
- Modify: `pyproject.toml` (add dependency)
- Create: `vram_mcp/nvml.py`
- Test: `tests/test_nvml.py`

**Interfaces:**
- Produces: `nvml_memory(device_index: int = 0, *, nvml=None) -> dict | None` → `{"total_mb","used_mb","free_mb","reserved_mb","uuid"}` or `None`.
- Produces: `nvml_processes(device_index: int = 0, *, nvml=None) -> list[dict]` → `[{"pid","size_mb","kind"}]`, `kind` in `{"compute","graphics"}`, `[]` on failure.
- Produces: `nvml_busy(pid: int, device_index: int = 0, *, nvml=None) -> bool | None`.

- [ ] **Step 1: Add the new dependency**

Edit `pyproject.toml`'s `dependencies` list:

```toml
dependencies = [
    "mcp>=1.2.0",
    "requests>=2.31",
    "nvidia-ml-py>=12.560",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_nvml.py`:

```python
"""Tests for vram_mcp.nvml — pure, no real GPU/driver needed."""

from vram_mcp import nvml as nvml_mod


class FakeNVMLError(Exception):
    pass


class _Mem:
    def __init__(self, total, used, free, reserved):
        self.total, self.used, self.free, self.reserved = total, used, free, reserved


class _Proc:
    def __init__(self, pid, used_gpu_memory):
        self.pid = pid
        self.usedGpuMemory = used_gpu_memory


class _UtilSample:
    def __init__(self, pid, sm_util):
        self.pid = pid
        self.smUtil = sm_util


class FakeNvml:
    """Stand-in for pynvml: canned responses, or raise NVMLError on demand."""
    NVMLError = FakeNVMLError

    def __init__(self, *, fail_init=False, fail_handle=False,
                 memory=None, compute_procs=None, graphics_procs=None,
                 util_samples=None, fail_memory=False, fail_compute=False,
                 fail_graphics=False, fail_utilization=False):
        self.fail_init = fail_init
        self.fail_handle = fail_handle
        self._memory = memory
        self._compute = compute_procs or []
        self._graphics = graphics_procs or []
        self._util = util_samples or []
        self.fail_memory = fail_memory
        self.fail_compute = fail_compute
        self.fail_graphics = fail_graphics
        self.fail_utilization = fail_utilization
        self.shutdown_called = False
        self.nvmlMemory_v2 = "v2-marker"

    def nvmlInit(self):
        if self.fail_init:
            raise FakeNVMLError("no driver")

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, index):
        if self.fail_handle:
            raise FakeNVMLError("no such device")
        return f"handle-{index}"

    def nvmlDeviceGetMemoryInfo(self, handle, version):
        if self.fail_memory:
            raise FakeNVMLError("memory unavailable")
        assert version == self.nvmlMemory_v2
        return self._memory

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-fake-uuid"

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle):
        if self.fail_compute:
            raise FakeNVMLError("not supported")
        return self._compute

    def nvmlDeviceGetGraphicsRunningProcesses_v3(self, handle):
        if self.fail_graphics:
            raise FakeNVMLError("not supported")
        return self._graphics

    def nvmlDeviceGetProcessUtilization(self, handle, timestamp):
        if self.fail_utilization:
            raise FakeNVMLError("not supported")
        return self._util


def test_nvml_memory_reports_v2_fields():
    fake = FakeNvml(memory=_Mem(total=24 * 1024**3, used=10 * 1024**3,
                                free=14 * 1024**3, reserved=250 * 1024**2))
    result = nvml_mod.nvml_memory(nvml=fake)
    assert result == {
        "total_mb": 24576, "used_mb": 10240, "free_mb": 14336,
        "reserved_mb": 250, "uuid": "GPU-fake-uuid",
    }
    assert fake.shutdown_called is True


def test_nvml_memory_none_when_init_fails():
    assert nvml_mod.nvml_memory(nvml=FakeNvml(fail_init=True)) is None


def test_nvml_memory_none_when_handle_fails():
    assert nvml_mod.nvml_memory(nvml=FakeNvml(fail_handle=True)) is None


def test_nvml_memory_none_when_query_fails_but_still_shuts_down():
    fake = FakeNvml(fail_memory=True)
    assert nvml_mod.nvml_memory(nvml=fake) is None
    assert fake.shutdown_called is True


def test_nvml_processes_merges_compute_and_graphics_dedup():
    fake = FakeNvml(
        compute_procs=[_Proc(100, 500 * 1024**2), _Proc(200, None)],
        graphics_procs=[_Proc(200, 300 * 1024**2), _Proc(300, 50 * 1024**2)],
    )
    result = nvml_mod.nvml_processes(nvml=fake)
    assert result == [
        {"pid": 100, "size_mb": 500, "kind": "compute"},
        {"pid": 200, "size_mb": None, "kind": "compute"},
        {"pid": 300, "size_mb": 50, "kind": "graphics"},
    ]


def test_nvml_processes_partial_failure_still_returns_other_list():
    fake = FakeNvml(fail_compute=True, graphics_procs=[_Proc(300, 50 * 1024**2)])
    result = nvml_mod.nvml_processes(nvml=fake)
    assert result == [{"pid": 300, "size_mb": 50, "kind": "graphics"}]


def test_nvml_processes_empty_when_unavailable():
    assert nvml_mod.nvml_processes(nvml=FakeNvml(fail_init=True)) == []


def test_nvml_busy_true_when_sample_has_positive_sm_util():
    fake = FakeNvml(util_samples=[_UtilSample(111, 0), _UtilSample(222, 40)])
    assert nvml_mod.nvml_busy(222, nvml=fake) is True


def test_nvml_busy_false_when_sample_present_but_zero():
    fake = FakeNvml(util_samples=[_UtilSample(111, 0)])
    assert nvml_mod.nvml_busy(111, nvml=fake) is False


def test_nvml_busy_none_when_pid_not_in_samples():
    fake = FakeNvml(util_samples=[_UtilSample(111, 40)])
    assert nvml_mod.nvml_busy(999, nvml=fake) is None


def test_nvml_busy_none_when_unavailable():
    assert nvml_mod.nvml_busy(111, nvml=FakeNvml(fail_init=True)) is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_nvml.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vram_mcp.nvml'`

- [ ] **Step 4: Write the implementation**

Create `vram_mcp/nvml.py`:

```python
"""NVML-based VRAM/process telemetry via nvidia-ml-py.

Pure module: the real ``pynvml`` (nvidia-ml-py) is injected as the ``nvml``
parameter, defaulting to the real package, so tests never need a real GPU or
driver. Every function degrades to ``None``/``[]`` on any ``nvml.NVMLError``
(no driver, non-NVIDIA GPU, unsupported call) — never raises to the caller.
"""

from __future__ import annotations

from typing import Optional

_BYTES_PER_MB = 1024 * 1024


def _b2mb(value) -> Optional[int]:
    if value is None:
        return None
    return int(value) // _BYTES_PER_MB


def _with_device(nvml, device_index, fn, default):
    """Init NVML, get the device handle, run ``fn(nvml, handle)``, always shut down.

    Returns ``default`` on any ``nvml.NVMLError`` from init, handle lookup, or
    ``fn`` itself.
    """
    try:
        nvml.nvmlInit()
    except nvml.NVMLError:
        return default
    try:
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        except nvml.NVMLError:
            return default
        try:
            return fn(nvml, handle)
        except nvml.NVMLError:
            return default
    finally:
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass


def nvml_memory(device_index: int = 0, *, nvml=None) -> Optional[dict]:
    """Accurate VRAM total/used/free/reserved + stable UUID for one GPU.

    Returns ``{"total_mb","used_mb","free_mb","reserved_mb","uuid"}``, or
    ``None`` if NVML is unavailable / the device index doesn't exist. Uses
    ``nvmlMemory_v2`` for the ``reserved`` (driver-reserved) field a plain
    ``nvidia-smi`` query can't give — v1's ``used`` conflates allocated and
    reserved memory.
    """
    if nvml is None:
        import pynvml as nvml

    def _query(nvml, handle):
        mem = nvml.nvmlDeviceGetMemoryInfo(handle, nvml.nvmlMemory_v2)
        return {
            "total_mb": _b2mb(mem.total),
            "used_mb": _b2mb(mem.used),
            "free_mb": _b2mb(mem.free),
            "reserved_mb": _b2mb(mem.reserved),
            "uuid": nvml.nvmlDeviceGetUUID(handle),
        }

    return _with_device(nvml, device_index, _query, None)


def nvml_processes(device_index: int = 0, *, nvml=None) -> list[dict]:
    """Every process holding VRAM on this GPU: compute AND graphics contexts.

    Returns ``[{"pid","size_mb","kind"}, ...]``; ``kind`` is ``"compute"`` or
    ``"graphics"``. ``size_mb`` is ``None`` when the driver doesn't report
    per-process memory (observed on Windows/WDDM — the PID/kind are still
    useful even without a size). A PID present in both lists is reported
    once, tagged ``"compute"``. The compute and graphics queries fail
    independently — if one is unsupported, the other's results still come
    back. ``[]`` if NVML is unavailable entirely.
    """
    if nvml is None:
        import pynvml as nvml

    def _query(nvml, handle):
        out: list[dict] = []
        seen: set[int] = set()
        try:
            compute = nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except nvml.NVMLError:
            compute = []
        for p in compute:
            out.append({"pid": p.pid, "size_mb": _b2mb(p.usedGpuMemory), "kind": "compute"})
            seen.add(p.pid)
        try:
            graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses_v3(handle)
        except nvml.NVMLError:
            graphics = []
        for p in graphics:
            if p.pid in seen:
                continue
            out.append({"pid": p.pid, "size_mb": _b2mb(p.usedGpuMemory), "kind": "graphics"})
        return out

    return _with_device(nvml, device_index, _query, [])


def nvml_busy(pid: int, device_index: int = 0, *, nvml=None) -> Optional[bool]:
    """Was ``pid`` doing GPU compute recently (windowed, not point-in-time)?

    Uses ``nvmlDeviceGetProcessUtilization(handle, 0)`` — ``0`` returns every
    sample in NVML's own short internal buffer, not a single instant, so a
    brief idle gap between generated tokens doesn't read as "idle" the way a
    single-shot snapshot could. Returns ``None`` (undetermined) if NVML is
    unavailable, or if ``pid`` has no sample in the buffer at all — never
    guesses ``False`` for a PID we simply have no data on.
    """
    if nvml is None:
        import pynvml as nvml

    def _query(nvml, handle):
        for s in nvml.nvmlDeviceGetProcessUtilization(handle, 0):
            if s.pid == pid:
                return s.smUtil > 0
        return None

    return _with_device(nvml, device_index, _query, None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_nvml.py -q`
Expected: `12 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml vram_mcp/nvml.py tests/test_nvml.py
git commit -m "feat: NVML wrapper for accurate VRAM, all-process visibility, windowed busy detection"
```

---

### Task 2: Ollama PID correlation (`ollama_correlate.py`)

**Files:**
- Create: `vram_mcp/ollama_correlate.py`
- Test: `tests/test_ollama_correlate.py`

**Interfaces:**
- Produces: `find_pid_for_model(model_name: str, *, list_processes=None, manifests_root: Path | None = None) -> int | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ollama_correlate.py`:

```python
"""Tests for vram_mcp.ollama_correlate — pure except real tmp_path manifests."""

import json

from vram_mcp import ollama_correlate as oc

_REAL_DIGEST = "a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"


def _write_manifest(root, namespace, name, tag, model_digest):
    path = root / namespace / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": f"sha256:{model_digest}"},
            {"mediaType": "application/vnd.ollama.image.template",
             "digest": "sha256:deadbeef"},
        ],
    }))


def test_extract_model_digest_from_real_cmdline():
    cmdline = (
        r"C:\Ollama\llama-server.exe --model "
        rf"C:\Users\x\.ollama\models\blobs\sha256-{_REAL_DIGEST} --port 12288"
    )
    assert oc._extract_model_digest(cmdline) == _REAL_DIGEST


def test_extract_model_digest_missing_flag_returns_none():
    assert oc._extract_model_digest("llama-server.exe --port 12288") is None


def test_extract_model_digest_short_hash_not_matched():
    # A malformed/truncated digest (not 64 hex chars) is never extracted --
    # never guess a partial match.
    assert oc._extract_model_digest("llama-server.exe --model /x/sha256-abc123") is None


def test_resolve_tag_for_digest_library_namespace(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "library", "qwen3", "8b", "abc123")
    assert oc._resolve_tag_for_digest("abc123", manifests) == "qwen3:8b"


def test_resolve_tag_for_digest_non_library_namespace(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "someuser", "mymodel", "latest", "abc123")
    assert oc._resolve_tag_for_digest("abc123", manifests) == "someuser/mymodel:latest"


def test_resolve_tag_for_digest_no_match_returns_none(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "library", "qwen3", "8b", "abc123")
    assert oc._resolve_tag_for_digest("nonexistent", manifests) is None


def test_resolve_tag_for_digest_missing_dir_returns_none(tmp_path):
    assert oc._resolve_tag_for_digest("abc123", tmp_path / "nope") is None


def test_find_pid_for_model_matches_correlated_tag(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "library", "qwen3", "8b", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 43176, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} --port 1"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid == 43176


def test_find_pid_for_model_no_matching_process_returns_none(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "library", "qwen3", "8b", _REAL_DIGEST)
    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=lambda: [], manifests_root=manifests
    )
    assert pid is None


def test_find_pid_for_model_digest_mismatch_returns_none(tmp_path):
    manifests = tmp_path / "manifests" / "registry.ollama.ai"
    _write_manifest(manifests, "library", "qwen3", "8b", _REAL_DIGEST)
    other_digest = "b" * 64

    def fake_list_processes():
        return [{"pid": 1, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{other_digest}"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ollama_correlate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vram_mcp.ollama_correlate'`

- [ ] **Step 3: Write the implementation**

Create `vram_mcp/ollama_correlate.py`:

```python
"""Map an Ollama model name to its ``llama-server`` runner PID.

Ollama spawns one ``llama-server`` OS subprocess per loaded model but exposes
no PID in its own API (``/api/ps`` has no ``pid`` field, confirmed against
Ollama 0.31.1). The only correlation path: read the runner's ``--model
<path>`` command-line argument (a blob file named ``sha256-<digest>``), then
search Ollama's on-disk manifests for the one whose model-layer digest
matches, which reveals the model's tag.

This is inherently undocumented and version-dependent. Every function
degrades to ``None`` on any parse/lookup failure rather than guessing, so a
future Ollama layout change only turns a model's ``busy`` signal into
"unknown," never breaks anything else.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

_MODEL_FLAG_RE = re.compile(r"--model\s+(\S+)")
_BLOB_DIGEST_RE = re.compile(r"sha256-([0-9a-f]{64})", re.IGNORECASE)
_MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _default_manifests_root() -> Path:
    models_dir = os.environ.get("OLLAMA_MODELS")
    if models_dir:
        return Path(models_dir) / "manifests"
    return Path.home() / ".ollama" / "models" / "manifests"


def _list_llama_server_processes_windows(timeout: int = 5) -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via one ``wmic`` call. ``[]`` on failure."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='llama-server.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, OSError):
        return []
    out = []
    lines = [ln.rstrip() for ln in result.stdout.splitlines() if ln.strip()]
    for line in lines[1:]:  # skip the header row
        parts = line.rsplit(None, 1)  # PID is the trailing whitespace-separated token
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        out.append({"pid": int(parts[1]), "cmdline": parts[0].strip()})
    return out


def _list_llama_server_processes_posix(timeout: int = 5) -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via ``ps``. ``[]`` on failure."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, OSError):
        return []
    out = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if "llama-server" not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        out.append({"pid": int(parts[0]), "cmdline": parts[1]})
    return out


def _list_llama_server_processes(timeout: int = 5) -> list[dict]:
    if sys.platform == "win32":
        return _list_llama_server_processes_windows(timeout)
    return _list_llama_server_processes_posix(timeout)


def _extract_model_digest(cmdline: str) -> Optional[str]:
    """Pull the sha256 hex digest out of a runner's ``--model <blob-path>`` arg.

    Requires exactly 64 hex characters (a real sha256 digest) — a shorter or
    malformed match is never accepted, so this never returns a partial guess.
    """
    m = _MODEL_FLAG_RE.search(cmdline)
    if not m:
        return None
    blob_match = _BLOB_DIGEST_RE.search(m.group(1))
    return blob_match.group(1).lower() if blob_match else None


def _resolve_tag_for_digest(digest: str, manifests_root: Path) -> Optional[str]:
    """Search every manifest file for one whose model layer matches ``digest``.

    A manifest's path (relative to ``manifests_root``) is
    ``<namespace>/<name>/<tag>``; the tag reported is ``name:tag`` for the
    default ``library`` namespace (matching Ollama's own ``/api/ps`` naming),
    or ``namespace/name:tag`` otherwise.
    """
    if not manifests_root.is_dir():
        return None
    target = f"sha256:{digest}"
    for manifest_path in manifests_root.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = doc.get("layers", [])
        match = any(
            layer.get("mediaType") == _MODEL_LAYER_MEDIA_TYPE and layer.get("digest") == target
            for layer in layers
        )
        if not match:
            continue
        rel = manifest_path.relative_to(manifests_root).parts
        if len(rel) < 3:
            continue
        tag = rel[-1]
        name = rel[-2]
        namespace = rel[-3] if len(rel) >= 4 else "library"
        return f"{name}:{tag}" if namespace == "library" else f"{namespace}/{name}:{tag}"
    return None


def find_pid_for_model(
    model_name: str, *,
    list_processes=None,
    manifests_root: Optional[Path] = None,
) -> Optional[int]:
    """The OS PID of the ``llama-server`` runner currently serving ``model_name``.

    ``None`` if Ollama isn't running that model, or if correlation fails for
    any reason (unexpected command-line shape, manifest missing/unparsable).
    """
    list_processes = list_processes or _list_llama_server_processes
    manifests_root = manifests_root or _default_manifests_root()
    for proc in list_processes():
        digest = _extract_model_digest(proc["cmdline"])
        if not digest:
            continue
        if _resolve_tag_for_digest(digest, manifests_root) == model_name:
            return proc["pid"]
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ollama_correlate.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/ollama_correlate.py tests/test_ollama_correlate.py
git commit -m "feat: correlate an Ollama model tag to its llama-server runner PID"
```

---

### Task 3: Shared claim ledger (`claims.py`)

**Files:**
- Create: `vram_mcp/claims.py`
- Test: `tests/test_claims.py`

**Interfaces:**
- Produces: `claim(model, owner, purpose, ttl_seconds=3600, *, path=None, now_fn=...) -> {"claim_id", "expires_at"}`
- Produces: `renew(claim_id, ttl_seconds=None, *, path=None, now_fn=...) -> {"ok", "expires_at"?}`
- Produces: `release(claim_id, *, path=None) -> {"ok"}`
- Produces: `list_claims(model=None, *, path=None, now_fn=...) -> list[dict]` — each dict has `claim_id, model, owner, purpose, claimed_at, renewed_at, ttl_seconds, expires_at`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claims.py`:

```python
"""Tests for vram_mcp.claims — real temp-dir file I/O (atomicity is the point)."""

from datetime import datetime, timedelta, timezone

import pytest

from vram_mcp import claims


def _clock(start):
    """A controllable now_fn: starts at `start`, advances via .tick(seconds)."""
    state = {"now": start}

    def now_fn():
        return state["now"]

    def tick(seconds):
        state["now"] = state["now"] + timedelta(seconds=seconds)

    now_fn.tick = tick
    return now_fn


def test_claim_creates_and_list_claims_returns_it(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    result = claims.claim("llama3.2", "retro-repo", "narration",
                          ttl_seconds=3600, path=path, now_fn=now_fn)
    assert "claim_id" in result
    assert result["expires_at"] == "2026-07-13T19:00:00Z"

    active = claims.list_claims(path=path, now_fn=now_fn)
    assert len(active) == 1
    assert active[0]["model"] == "llama3.2"
    assert active[0]["owner"] == "retro-repo"
    assert active[0]["purpose"] == "narration"


def test_list_claims_filters_by_model(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    claims.claim("qwen3:8b", "b", "y", path=path, now_fn=now_fn)

    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 2
    only_qwen = claims.list_claims("qwen3:8b", path=path, now_fn=now_fn)
    assert len(only_qwen) == 1
    assert only_qwen[0]["owner"] == "b"


def test_claim_expires_after_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(30)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1

    now_fn.tick(31)  # 61s total -> past the 60s ttl
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_renew_extends_expiry_with_original_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(50)
    result = claims.renew(created["claim_id"], path=path, now_fn=now_fn)
    assert result["ok"] is True
    assert result["expires_at"] == "2026-07-13T18:01:50Z"  # now(18:00:50) + 60s

    now_fn.tick(55)  # 105s since creation, 55s since renew -> still active
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1


def test_renew_with_new_ttl_overrides(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    result = claims.renew(created["claim_id"], ttl_seconds=7200, path=path, now_fn=now_fn)
    assert result["expires_at"] == "2026-07-13T20:00:00Z"


def test_renew_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.renew("nonexistent-id", path=path) == {"ok": False}


def test_release_removes_claim(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    created = claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert claims.release(created["claim_id"], path=path) == {"ok": True}
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_release_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.release("nonexistent", path=path) == {"ok": False}


def test_multiple_claims_same_model_independent(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    claims.claim("llama3.2", "session-a", "reason-a", path=path, now_fn=now_fn)
    claims.claim("llama3.2", "session-b", "reason-b", path=path, now_fn=now_fn)
    active = claims.list_claims("llama3.2", path=path, now_fn=now_fn)
    assert {c["owner"] for c in active} == {"session-a", "session-b"}


def test_sequential_claims_both_persist(tmp_path):
    """Each claim()/release() call round-trips through the lock cleanly --
    a stale lock from a prior call never blocks the next one."""
    path = tmp_path / "claims.json"
    now_fn = _clock(datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc))
    claims.claim("a", "x", "p", path=path, now_fn=now_fn)
    claims.claim("b", "y", "p", path=path, now_fn=now_fn)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 2


def test_locked_raises_timeout_if_lock_file_already_held(tmp_path):
    path = tmp_path / "claims.json"
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os_open_lock(lock_path)
    try:
        with pytest.raises(TimeoutError):
            with claims._locked(path, timeout=0.2, poll=0.05):
                pass
    finally:
        os_close_lock(fd, lock_path)


def os_open_lock(lock_path):
    import os
    return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)


def os_close_lock(fd, lock_path):
    import os
    os.close(fd)
    os.remove(lock_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_claims.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vram_mcp.claims'`

- [ ] **Step 3: Write the implementation**

Create `vram_mcp/claims.py`:

```python
"""Shared, file-based attribution ledger.

Every ``vram-mcp`` session runs its own independent subprocess (no shared
server), so cross-session attribution needs a shared file on disk:
``~/.cache/vram-mcp/claims.json`` by default. Writes are atomic (temp file +
``os.replace``) and serialized with a sibling lock file so concurrent
sessions can't corrupt each other's writes. A claim expires via TTL — no
explicit release is required, so a session that crashes without cleanup
doesn't leave a permanently-stuck claim.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

_DEFAULT_PATH = Path.home() / ".cache" / "vram-mcp" / "claims.json"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@contextmanager
def _locked(path: Path, timeout: float = 5.0, poll: float = 0.05):
    """Serialize access to ``path`` via a sibling ``.lock`` file.

    Uses ``os.open`` with ``O_CREAT | O_EXCL`` — atomic file creation that
    fails if the lock already exists, cross-platform (Windows and POSIX both
    honor ``O_EXCL``). Raises ``TimeoutError`` if the lock can't be acquired
    within ``timeout`` seconds (another process holds it).
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _load(path: Path) -> dict:
    if not path.exists():
        return {"claims": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"claims": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _is_active(record: dict, now: datetime) -> bool:
    return now < _parse_iso(record["expires_at"])


def claim(
    model: str, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Register that ``owner`` is using ``model`` for ``purpose``.

    Returns ``{"claim_id", "expires_at"}``. Multiple claims may exist for the
    same model at once (different owners); each is independent.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    expires_at = now + timedelta(seconds=ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex,
        "model": model, "owner": owner, "purpose": purpose,
        "claimed_at": _iso(now), "renewed_at": _iso(now),
        "ttl_seconds": ttl_seconds, "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}


def renew(
    claim_id: str, ttl_seconds: Optional[int] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Extend a claim's expiry to ``now + ttl_seconds`` (or its original TTL).

    Returns ``{"ok": False}`` if no claim with ``claim_id`` exists.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    with _locked(path):
        data = _load(path)
        for record in data["claims"]:
            if record["claim_id"] == claim_id:
                ttl = ttl_seconds if ttl_seconds is not None else record["ttl_seconds"]
                record["ttl_seconds"] = ttl
                record["renewed_at"] = _iso(now)
                record["expires_at"] = _iso(now + timedelta(seconds=ttl))
                _save(path, data)
                return {"ok": True, "expires_at": record["expires_at"]}
    return {"ok": False}


def release(claim_id: str, *, path: Optional[Path] = None) -> dict:
    """Remove a claim immediately (before its TTL expires)."""
    path = path or _DEFAULT_PATH
    with _locked(path):
        data = _load(path)
        before = len(data["claims"])
        data["claims"] = [r for r in data["claims"] if r["claim_id"] != claim_id]
        found = len(data["claims"]) != before
        if found:
            _save(path, data)
    return {"ok": found}


def list_claims(
    model: Optional[str] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> list[dict]:
    """Every currently-ACTIVE claim, optionally filtered to one model.

    Expired claims are silently excluded (not deleted — ``claim()``/
    ``release()`` are the only writers; a read never needs the lock since
    writes are atomic, so a reader always sees a complete file, old or new).
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    active = [r for r in _load(path)["claims"] if _is_active(r, now)]
    if model is not None:
        active = [r for r in active if r["model"] == model]
    return active
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_claims.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/claims.py tests/test_claims.py
git commit -m "feat: shared, TTL-based, file-locked claim ledger for cross-session attribution"
```

---

### Task 4: Status enrichment (`core.py` read path)

**Files:**
- Modify: `vram_mcp/core.py`
- Modify: `tests/test_core.py` (update 2 existing assertions, add new tests)

**Interfaces:**
- Consumes (from Tasks 1–3, as injected callables — `core.py` never imports `nvml`/`ollama_correlate`/`claims` directly): a `list_claims_fn(model: str) -> list[dict]` shaped like `claims.list_claims`; a `find_pid_fn(model: str) -> int | None` shaped like `ollama_correlate.find_pid_for_model`; a `busy_fn(pid: int) -> bool | None` shaped like `nvml.nvml_busy`; a `nvml_processes_fn() -> list[dict]` shaped like `nvml.nvml_processes`.
- Produces: `attach_claims(loaded, list_claims_fn) -> list[dict]`; `attach_busy(loaded, find_pid_fn, busy_fn) -> tuple[list[dict], set[int]]`; `other_processes(nvml_processes_fn, exclude_pids) -> list[dict]`; updated `combined_status(gpu_status_fn, ollama, *, list_claims_fn=None, find_pid_fn=None, busy_fn=None, nvml_processes_fn=None) -> dict` (all-optional kwargs — existing callers that pass only the first two positional args see the same base shape, now with `total_size_mb`/`offloaded_to_cpu` added to each loaded-model dict).

- [ ] **Step 1: Write the failing tests**

In `tests/test_core.py`, update the two existing assertions that check the loaded-model dict shape:

```python
def test_combined_status():
    models = [
        {"name": "big", "size": gb_bytes(8), "size_vram": gb_bytes(8),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    status = core.combined_status(gpu_fn_const(6000), FakeOllama(models))
    assert status["free_mb"] == 6000
    assert len(status["gpus"]) == 1
    assert status["loaded"] == [
        {"name": "big", "size_vram_mb": 8192, "total_size_mb": 8192,
         "offloaded_to_cpu": False, "expires_at": "2026-07-11T10:00:00Z"},
    ]


def test_combined_status_no_gpu():
    status = core.combined_status(gpu_fn_const(None), FakeOllama([]))
    assert status["gpus"] == []
    assert status["free_mb"] is None
    assert status["loaded"] == []
```

Then add new tests to the same file:

```python
# ---- offload detection -------------------------------------------------------

def test_loaded_models_detects_cpu_offload():
    models = [
        {"name": "partial", "size": gb_bytes(10), "size_vram": gb_bytes(6),
         "expires_at": None},
    ]
    status = core.combined_status(gpu_fn_const(4000), FakeOllama(models))
    entry = status["loaded"][0]
    assert entry["total_size_mb"] == 10240
    assert entry["size_vram_mb"] == 6144
    assert entry["offloaded_to_cpu"] is True


# ---- attach_claims ------------------------------------------------------------

def test_attach_claims_adds_claims_list_per_model():
    loaded = [{"name": "m1"}, {"name": "m2"}]

    def list_claims_fn(model):
        return [{"owner": "x"}] if model == "m1" else []

    result = core.attach_claims(loaded, list_claims_fn)
    assert result[0]["claims"] == [{"owner": "x"}]
    assert result[1]["claims"] == []


# ---- attach_busy ----------------------------------------------------------------

def test_attach_busy_resolves_pid_and_busy_flag():
    loaded = [{"name": "m1"}, {"name": "m2"}]

    def find_pid_fn(model):
        return 123 if model == "m1" else None

    def busy_fn(pid):
        return pid == 123

    result, resolved = core.attach_busy(loaded, find_pid_fn, busy_fn)
    assert result[0]["busy"] is True
    assert result[1]["busy"] is None  # no pid -> never call busy_fn's real signal
    assert resolved == {123}


# ---- other_processes ------------------------------------------------------------

def test_other_processes_excludes_known_ollama_pids():
    def nvml_processes_fn():
        return [
            {"pid": 100, "size_mb": 500, "kind": "compute"},
            {"pid": 200, "size_mb": 300, "kind": "graphics"},
        ]

    result = core.other_processes(nvml_processes_fn, exclude_pids={100})
    assert result == [{"pid": 200, "size_mb": 300, "kind": "graphics"}]


# ---- combined_status full wiring -------------------------------------------------

def test_combined_status_full_wiring():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4),
              "expires_at": None}]
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        list_claims_fn=lambda model: [{"owner": "x"}],
        find_pid_fn=lambda model: 555,
        busy_fn=lambda pid: True,
        nvml_processes_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "kind": "compute"},
            {"pid": 999, "size_mb": 100, "kind": "graphics"},
        ],
    )
    entry = status["loaded"][0]
    assert entry["claims"] == [{"owner": "x"}]
    assert entry["busy"] is True
    # pid 555 IS the "m1" runner -> excluded from other_processes; pid 999 stays.
    assert status["other_processes"] == [{"pid": 999, "size_mb": 100, "kind": "graphics"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -q`
Expected: FAIL — the two updated tests fail on the current (unenriched) dict shape; the new tests fail with `AttributeError: module 'vram_mcp.core' has no attribute 'attach_claims'` etc.

- [ ] **Step 3: Write the implementation**

In `vram_mcp/core.py`, replace `_loaded_models` and `combined_status`, and add the three new functions (insert after `combined_status`):

```python
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
```

(`_bytes_to_mb`, `_gpu`, `Callable` are already imported/defined earlier in `core.py` — no new imports needed for this task.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -q`
Expected: all pass (existing + new tests in this file)

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/core.py tests/test_core.py
git commit -m "feat: core.py enriches status with offload detection, claims, busy signal, other processes"
```

---

### Task 5: Eviction protection (`core.py` write path)

**Files:**
- Modify: `vram_mcp/core.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: same three injected callables as Task 4 (`list_claims_fn`, `find_pid_fn`, `busy_fn`).
- Produces: `is_protected(model_name, list_claims_fn, find_pid_fn, busy_fn) -> tuple[bool, dict]`; updated `ensure_free(target_gb, gpu_status_fn, ollama, settle=0.0, sleep=time.sleep, *, list_claims_fn=None, find_pid_fn=None, busy_fn=None, force=False) -> dict` — result gains a `"declined"` key (list of skipped-protected-model details) alongside the existing `ok/already_free/free_mb/unloaded/target_mb`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core.py`:

```python
# ---- is_protected -----------------------------------------------------------

def test_is_protected_true_when_active_claim_exists():
    protected, detail = core.is_protected(
        "m", list_claims_fn=lambda name: [{"owner": "x"}],
        find_pid_fn=lambda name: None, busy_fn=lambda pid: None,
    )
    assert protected is True
    assert detail["claims"] == [{"owner": "x"}]


def test_is_protected_true_when_busy():
    protected, detail = core.is_protected(
        "m", list_claims_fn=lambda name: [], find_pid_fn=lambda name: 123,
        busy_fn=lambda pid: True,
    )
    assert protected is True
    assert detail["busy"] is True


def test_is_protected_false_when_unclaimed_and_idle():
    protected, _ = core.is_protected(
        "m", list_claims_fn=lambda name: [], find_pid_fn=lambda name: 123,
        busy_fn=lambda pid: False,
    )
    assert protected is False


def test_is_protected_false_when_no_pid_and_no_claim():
    protected, detail = core.is_protected(
        "m", list_claims_fn=lambda name: [], find_pid_fn=lambda name: None,
        busy_fn=lambda pid: True,  # never called: no pid to check
    )
    assert protected is False
    assert detail["busy"] is None


# ---- ensure_free protection ---------------------------------------------------

def test_ensure_free_skips_protected_model_and_reports_declined():
    models = [
        {"name": "protected", "size_vram": gb_bytes(10), "expires_at": None},
        {"name": "free-game", "size_vram": gb_bytes(6), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 7000])

    result = core.ensure_free(
        6, gpu_fn, ollama, sleep=lambda *_: None,
        list_claims_fn=lambda name: [{"owner": "other"}] if name == "protected" else [],
        find_pid_fn=lambda name: None,
        busy_fn=lambda pid: None,
    )
    assert ollama.unloaded == ["free-game"]  # "protected" skipped despite being largest
    assert result["ok"] is True
    assert len(result["declined"]) == 1
    assert result["declined"][0]["name"] == "protected"


def test_ensure_free_force_bypasses_protection():
    models = [{"name": "protected", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])

    result = core.ensure_free(
        8, gpu_fn, ollama, sleep=lambda *_: None, force=True,
        list_claims_fn=lambda name: [{"owner": "other"}],
        find_pid_fn=lambda name: None, busy_fn=lambda pid: None,
    )
    assert ollama.unloaded == ["protected"]
    assert result["declined"] == []


def test_ensure_free_protection_noop_when_fns_not_provided():
    """Existing callers that don't wire claims/busy see unchanged behavior."""
    models = [{"name": "a", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert ollama.unloaded == ["a"]
    assert result["declined"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -q`
Expected: FAIL — `is_protected` tests fail with `AttributeError`; `ensure_free` protection tests fail because `result["declined"]` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

In `vram_mcp/core.py`, add `is_protected` (after `other_processes`) and replace `ensure_free`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/core.py tests/test_core.py
git commit -m "feat: ensure_free skips claimed/busy models by default (force=True to override)"
```

---

### Task 6: Server tool surface + docs

**Files:**
- Modify: `vram_mcp/server.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `vram_mcp.claims` (Task 3), `vram_mcp.nvml` (Task 1), `vram_mcp.ollama_correlate.find_pid_for_model` (Task 2), `core.combined_status`/`core.is_protected`/`core.ensure_free` (Tasks 4–5).
- Produces (new MCP tools): `claim(model, owner, purpose, ttl_seconds=3600) -> dict`; `renew(claim_id, ttl_seconds=None) -> dict`; `release(claim_id) -> dict`; `list_claims(model=None) -> dict`.
- Produces (changed MCP tools): `vram_status()`/`list_loaded()` now include `claims`/`busy`/`total_size_mb`/`offloaded_to_cpu` per model and a top-level `other_processes`; `unload(model, force=False)`; `ensure_free(gb, force=False)`.

This task has no dedicated new test file — matching this project's existing convention (`server.py` has zero direct tests today; every tool is a thin wrapper whose logic is already covered by `test_core.py`/`test_claims.py`). Verification is a manual import + smoke-call, per Step 4 below.

- [ ] **Step 1: Add imports and wiring helpers**

In `vram_mcp/server.py`, after the existing imports (`from .ollama import OllamaClient`), add:

```python
from typing import Optional

from . import claims as _claims
from . import nvml as _nvml
from .ollama_correlate import find_pid_for_model as _find_pid_for_model
```

After the `_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)` line, add the wiring helpers `vram_status`/`ensure_free`/etc. will pass into `core`:

```python
def _list_claims_fn(model: str) -> list[dict]:
    return _claims.list_claims(model)


def _busy_fn(pid) -> Optional[bool]:
    return _nvml.nvml_busy(pid)


def _nvml_processes_fn() -> list[dict]:
    return _nvml.nvml_processes()
```

- [ ] **Step 2: Update `vram_status`, `list_loaded`, `unload`, `ensure_free`**

Replace the four existing tool functions in `vram_mcp/server.py`:

```python
@mcp.tool()
def vram_status() -> dict:
    """Report GPU VRAM and currently loaded Ollama models.

    Returns per-GPU totals, the list of resident models (each with claim
    attribution, a best-effort busy signal, and CPU-offload detection), every
    other VRAM-holding process on the GPU, the best free VRAM, and a
    human-readable ``summary``.
    """
    status = core.combined_status(
        gpu_status, _ollama,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
        nvml_processes_fn=_nvml_processes_fn,
    )
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"])
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}."
    )
    return status


@mcp.tool()
def list_loaded() -> dict:
    """List the models currently resident in VRAM, with claim/busy detail."""
    status = core.combined_status(
        gpu_status, _ollama,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
        nvml_processes_fn=_nvml_processes_fn,
    )
    loaded = status["loaded"]
    return {
        "loaded": loaded,
        "summary": f"{len(loaded)} model(s) loaded.",
    }


@mcp.tool()
def unload(model: str, force: bool = False) -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort
    ``busy`` signal — pass ``force=True`` to override.
    """
    if not force:
        protected, detail = core.is_protected(
            model, _list_claims_fn, _find_pid_for_model, _busy_fn,
        )
        if protected:
            return {
                "ok": False,
                "model": model,
                "protected": True,
                **detail,
                "summary": (
                    f"'{model}' is protected (claimed or busy); "
                    "pass force=True to override."
                ),
            }
    ok = _ollama.unload(model)
    return {
        "ok": ok,
        "model": model,
        "summary": (
            f"Unloaded '{model}'." if ok else f"Failed to unload '{model}'."
        ),
    }


@mcp.tool()
def ensure_free(gb: float, force: bool = False) -> dict:
    """Free VRAM until at least ``gb`` gigabytes are available.

    Unloads resident models largest-first until the target is met, skipping
    claimed/busy models by default (``force=True`` to override). Returns
    which models were unloaded, which were declined (protected), and whether
    the target was reached.
    """
    result = core.ensure_free(
        gb, gpu_status, _ollama, settle=0.5, force=force,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
    )
    declined_note = ""
    if result["declined"]:
        names = ", ".join(d["name"] for d in result["declined"])
        declined_note = f" Protected (force=True to override): {names}."
    if result["already_free"]:
        result["summary"] = (
            f"Already {_fmt_free(result['free_mb'])} free (target {gb} GB)."
        )
    elif result["ok"]:
        result["summary"] = (
            f"Freed VRAM to {_fmt_free(result['free_mb'])} "
            f"(target {gb} GB) by unloading: "
            f"{', '.join(result['unloaded']) or 'none'}."
        ) + declined_note
    else:
        result["summary"] = (
            f"Could not reach {gb} GB free "
            f"(now {_fmt_free(result['free_mb'])}); "
            f"unloaded: {', '.join(result['unloaded']) or 'none'}."
        ) + declined_note
    return result
```

- [ ] **Step 3: Add the four new tools**

Add after `ensure_free`, before `advise`:

```python
@mcp.tool()
def claim(model: str, owner: str, purpose: str, ttl_seconds: int = 3600) -> dict:
    """Declare that you're using ``model`` for ``purpose``.

    Lets other sessions see who's using a model and why before deciding to
    evict it. Renew before ``ttl_seconds`` elapses if still in use — an
    un-renewed claim simply expires, so a crashed session never leaves a
    permanently-stuck claim.
    """
    result = _claims.claim(model, owner, purpose, ttl_seconds)
    result["summary"] = (
        f"Claimed '{model}' for {owner} ({purpose}), expires {result['expires_at']}."
    )
    return result


@mcp.tool()
def renew(claim_id: str, ttl_seconds: Optional[int] = None) -> dict:
    """Extend an existing claim's expiry before it lapses."""
    result = _claims.renew(claim_id, ttl_seconds)
    result["summary"] = (
        f"Renewed, expires {result['expires_at']}." if result["ok"]
        else "No such claim (already expired or released?)."
    )
    return result


@mcp.tool()
def release(claim_id: str) -> dict:
    """Release a claim early, before its TTL would expire."""
    result = _claims.release(claim_id)
    result["summary"] = "Released." if result["ok"] else "No such claim."
    return result


@mcp.tool()
def list_claims(model: Optional[str] = None) -> dict:
    """See who's claiming what right now (all models, or one)."""
    active = _claims.list_claims(model)
    return {"claims": active, "summary": f"{len(active)} active claim(s)."}
```

- [ ] **Step 4: Manual smoke verification**

Run:

```bash
python -c "
from vram_mcp import server
print(server.claim.fn('llama3.2', 'smoke-test', 'verify', ttl_seconds=60))
print(server.list_claims.fn())
r = server.unload.fn('llama3.2')
print(r)
assert r['protected'] is True
print(server.release.fn(server.claim.fn('x', 'y', 'z')['claim_id']))
print(server.vram_status.fn())
"
```

Expected: no exceptions; `list_claims` shows the claim just created; `unload("llama3.2")` returns `protected: True` (it's claimed); `vram_status()` prints a dict whose `loaded`/`other_processes` keys are present (empty lists are fine if no model is actually loaded on this machine — the point is it runs without error). (FastMCP tool objects expose the original callable via `.fn` — confirm this attribute name against the installed `mcp` package if it differs; the smoke test's purpose is catching import/wiring errors, not exercising real GPU state.)

- [ ] **Step 5: Update the test suite + full run**

Run: `python -m pytest -q`
Expected: all tests across `test_nvml.py`, `test_ollama_correlate.py`, `test_claims.py`, `test_core.py`, `test_gpu.py`, `test_ollama.py` pass.

- [ ] **Step 6: Update README.md**

Add rows to the existing tools table and a short new section. In the `## Tools` table, add:

```markdown
| `claim(model, owner, purpose, ttl_seconds=3600)` | Declare you're using a model, so others see who/why before evicting it. |
| `renew(claim_id, ttl_seconds=None)` | Extend a claim before it expires. |
| `release(claim_id)` | Release a claim early. |
| `list_claims(model=None)` | See active claims (all models, or one). |
```

Update the `unload`/`ensure_free` rows to mention `force`:

```markdown
| `unload(model, force=False)` | Evict one model from VRAM now (`keep_alive=0`). Refuses if claimed/busy unless `force=True`. |
| `ensure_free(gb, force=False)` | Unload models largest-first until at least `gb` GB is free, skipping claimed/busy models unless `force=True`. |
```

Add a new section after `## Configuration`:

```markdown
## Multi-session coordination

Since every session runs its own `vram-mcp` process, coordination happens via:

- **Claims** — a shared, file-based ledger (`~/.cache/vram-mcp/claims.json`) recording who's using a model and why. Call `claim()` when you start relying on a model; `renew()` periodically if still in use. An un-renewed claim simply expires — no cleanup needed if your session ends unexpectedly.
- **Busy detection** — best-effort, via NVML's per-process GPU utilization (not point-in-time; reads a short recent window so brief gaps between tokens don't misread as idle). Requires no changes to how you call Ollama — it's entirely on vram-mcp's side.
- **Protection** — `unload()`/`ensure_free()` refuse to evict a model that's claimed OR busy, by default. Pass `force=True` when you've already decided it's worth it.

Requires the `nvidia-ml-py` dependency (installed automatically). Falls back gracefully — `claims`/`busy` report as empty/`null` — on non-NVIDIA GPUs or if NVML is unavailable.
```

- [ ] **Step 7: Commit**

```bash
git add vram_mcp/server.py README.md
git commit -m "feat: expose claim/renew/release/list_claims tools; protect unload/ensure_free by default"
```

---

## Self-Review

**Spec coverage:**
- Attribution (self-reported claim ledger) → Task 3 (`claims.py`) + Task 6 (`claim`/`renew`/`release`/`list_claims` tools). ✓
- Busy detection, windowed via NVML → Task 1 (`nvml_busy`). ✓
- All-process VRAM visibility → Task 1 (`nvml_processes`) + Task 4 (`other_processes`, exclusion of known Ollama PIDs). ✓
- PID→model correlation, isolated/fragile-by-design → Task 2 (`ollama_correlate.py`). ✓
- `nvmlDeviceGetMemoryInfo_v2`'s `reserved` field for a more accurate free-VRAM number → Task 1 (`nvml_memory`). ✓
- Protection = claim OR busy, not claim alone; `force` bypasses both → Task 5 (`is_protected`, `ensure_free`) + Task 6 (`unload`). ✓
- TTL-based claim expiry, crash-safe → Task 3 (`list_claims` excludes expired; no explicit cleanup). ✓
- UUID for stable per-GPU identity → Task 1 (`nvml_memory`'s `uuid` field). ✓
- UTC ISO 8601 timestamps → Task 3 (`_iso`/`_parse_iso`). ✓
- Free-form `owner` label → Task 3/6 (plain string, no parsing/validation beyond presence). ✓
- WDDM per-process memory gap handled gracefully (`size_mb: None`) → Task 1 (`nvml_processes`), verified empirically against this exact machine during design research. ✓
- New dependency in `pyproject.toml` → Task 1, Step 1. ✓
- Testing philosophy (pure modules, injectable, no real GPU/driver/Ollama needed) → Tasks 1–2 unit tests; Task 3 uses real `tmp_path` I/O deliberately (atomicity is the thing under test, matching the spec's own testing note). ✓
- Out-of-scope items (accounting mode, per-call wrapping, non-NVIDIA backends) — correctly absent from every task. ✓

**Placeholder scan:** no TBD/TODO; every step has complete, runnable code; no "add appropriate error handling" hand-waves — every degrade-to-`None`/`[]`/`False` path is written out explicitly with a stated reason.

**Type/naming consistency:** `claims` (plural, list) is used identically in `core.attach_claims`'s output key, `core.is_protected`'s `detail["claims"]`, and `server.py`'s tool outputs — this is a **deliberate refinement of the spec's wording**, which said singular "claim" (a summary). Plural avoids silently hiding a co-existing second claim on the same model and matches `is_protected`'s own shape, so nothing in the codebase uses two different names for the same concept. `list_claims_fn`/`find_pid_fn`/`busy_fn`/`nvml_processes_fn` keyword names are identical across Tasks 4, 5, and 6 wherever they're threaded through (`core.combined_status`, `core.ensure_free`, `core.is_protected`, and their call sites in `server.py`). `renew(claim_id, ttl_seconds=None)`'s signature matches between `claims.py` (Task 3) and its `server.py` tool wrapper (Task 6). `is_protected`'s parameter order (`model_name, list_claims_fn, find_pid_fn, busy_fn`) is identical at its one definition (Task 5) and its two call sites (`ensure_free` in Task 5, `unload` in Task 6).

**Scope check:** six tasks, each producing an independently testable deliverable, strictly sequential dependency chain (1/2/3 are independent of each other; 4 depends on the shape of 1–3's functions; 5 depends on 4; 6 depends on 3 and 5) — appropriately sized for one plan, matches the existing project's small-module scale.
