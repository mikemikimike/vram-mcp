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
