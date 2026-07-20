"""Tests for vram_mcp.procinfo — pure, injected readers."""
from vram_mcp import procinfo


def test_process_table_uses_nvml_sizes_when_present():
    # Linux/TCC: NVML gives real sizes; names come from the posix reader.
    nvml = lambda: [{"pid": 100, "size_mb": 8000, "kind": "compute"}]
    names = lambda pids: {100: {"name": "python", "cmdline": "python train.py"}}
    out = procinfo.process_table(nvml_processes=nvml, posix_name_reader=names)
    assert out == [{"pid": 100, "size_mb": 8000, "shared_mb": None,
                    "non_local_mb": None, "name": "python",
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
    win = lambda: [{"pid": 300, "size_mb": 2048, "shared_mb": 0,
                    "non_local_mb": 0, "name": "UnrealEditor.exe",
                    "cmdline": "UnrealEditor.exe Project.uproject"}]
    out = procinfo.process_table(nvml_processes=nvml, win_gpu_reader=win)
    assert out == [{"pid": 300, "size_mb": 2048, "shared_mb": 0,
                    "non_local_mb": 0, "name": "UnrealEditor.exe",
                    "cmdline": "UnrealEditor.exe Project.uproject", "kind": "compute"}]


def test_process_table_no_readers_returns_nvml_only_unnamed():
    nvml = lambda: [{"pid": 100, "size_mb": None, "kind": "compute"}]
    out = procinfo.process_table(nvml_processes=nvml)
    assert out == [{"pid": 100, "size_mb": None, "shared_mb": None,
                    "non_local_mb": None, "name": None,
                    "cmdline": None, "kind": "compute"}]


def test_process_table_merges_spill_fields():
    # Non Local usage = driver spilled VRAM into system RAM; must survive the merge.
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": None, "kind": "compute"}],
        win_gpu_reader=lambda: [
            {"pid": 7, "size_mb": 500, "shared_mb": 12, "non_local_mb": 300,
             "name": "a.exe", "cmdline": "a"},
        ],
    )
    assert rows[0]["non_local_mb"] == 300
    assert rows[0]["shared_mb"] == 12
    assert rows[0]["size_mb"] == 500


def test_process_table_defaults_spill_fields_to_none():
    # Linux/TCC has no perf counter — "unreported" must not read as "no spill".
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": 100, "kind": "compute"}],
    )
    assert rows[0]["shared_mb"] is None
    assert rows[0]["non_local_mb"] is None


def test_win_gpu_procs_parses_pipe_lines(monkeypatch):
    # pid|dedicated|shared|non-local|name|cmdline from the combined reader.
    fake = ("11924|15196000000|0|0|python.exe|python.exe train_lora_kg.py\n"
            "1336|1841000000|0|0|dwm.exe|dwm.exe\n")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: fake)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    out = {p["pid"]: p for p in procinfo.win_gpu_procs()}
    assert out[11924]["size_mb"] == 14492  # 15.196e9 // 1MB
    assert out[11924]["name"] == "python.exe"
    assert out[1336]["name"] == "dwm.exe"


def test_win_gpu_procs_parses_three_counters(monkeypatch):
    # One sample covers Dedicated + Shared + Non Local, so spill is visible.
    out = (
        "1328|948961280|104857600|0|dwm.exe|C:\\Windows\\dwm.exe\n"
        "28384|18229198848|2097152|1073741824|python.exe|python train.py --a b\n"
    )
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    rows = {r["pid"]: r for r in procinfo.win_gpu_procs()}
    assert rows[1328]["size_mb"] == 905
    assert rows[1328]["shared_mb"] == 100
    assert rows[1328]["non_local_mb"] == 0
    assert rows[28384]["size_mb"] == 17384  # floored, per bytes_to_mb
    assert rows[28384]["non_local_mb"] == 1024
    assert rows[28384]["name"] == "python.exe"


def test_win_gpu_procs_keeps_pipe_in_cmdline(monkeypatch):
    # cmdline is last so split("|", 5) leaves its own pipes intact.
    out = "42|1048576|0|0|sh.exe|sh -c 'a | b | c'\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    (row,) = procinfo.win_gpu_procs()
    assert row["cmdline"] == "sh -c 'a | b | c'"


def test_win_gpu_procs_skips_malformed_lines(monkeypatch):
    out = "not-a-pid|1|2|3|x|y\n7|1048576|0|0|a.exe|a\nshort|line\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    assert [r["pid"] for r in procinfo.win_gpu_procs()] == [7]


def test_win_gpu_procs_empty_off_windows(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "linux")
    assert procinfo.win_gpu_procs() == []


def test_win_gpu_procs_empty_on_reader_failure(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: None)
    assert procinfo.win_gpu_procs() == []
