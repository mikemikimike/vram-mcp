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
