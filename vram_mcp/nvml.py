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
