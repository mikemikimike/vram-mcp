"""Tests for vram_mcp.gpu — pure, no mcp / real GPU needed."""

import subprocess

from vram_mcp import gpu


FAKE_CSV = (
    "0, NVIDIA GeForce RTX 4090, 24564, 8192, 16372\n"
    "1, NVIDIA GeForce RTX 3060, 12288, 2048, 10240\n"
)


def test_gpu_status_parses_csv():
    gpus = gpu.gpu_status(runner=lambda cmd, timeout=5: FAKE_CSV)
    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "total_mb": 24564,
            "used_mb": 8192,
            "free_mb": 16372,
        },
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 3060",
            "total_mb": 12288,
            "used_mb": 2048,
            "free_mb": 10240,
        },
    ]


def test_gpu_status_skips_blank_lines():
    csv = "\n0, GPU0, 100, 40, 60\n\n"
    gpus = gpu.gpu_status(runner=lambda cmd, timeout=5: csv)
    assert len(gpus) == 1
    assert gpus[0]["free_mb"] == 60


def test_gpu_status_file_not_found_returns_empty():
    def boom(cmd, timeout=5):
        raise FileNotFoundError("nvidia-smi not found")

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_timeout_returns_empty():
    def boom(cmd, timeout=5):
        raise subprocess.TimeoutExpired(cmd, timeout)

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_nonzero_exit_returns_empty():
    def boom(cmd, timeout=5):
        raise subprocess.CalledProcessError(1, cmd)

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_parse_error_returns_empty():
    # Non-integer memory field -> unparseable -> [].
    csv = "0, GPU0, N/A, N/A, N/A\n"
    assert gpu.gpu_status(runner=lambda cmd, timeout=5: csv) == []


def test_gpu_status_wrong_column_count_returns_empty():
    csv = "0, GPU0, 100, 40\n"  # only 4 columns
    assert gpu.gpu_status(runner=lambda cmd, timeout=5: csv) == []


def test_gpu_status_default_runner_no_nvidia_smi(monkeypatch):
    # Force the real _run to look like nvidia-smi is missing.
    def boom(cmd, timeout=5):
        raise FileNotFoundError

    monkeypatch.setattr(gpu, "_run", boom)
    assert gpu.gpu_status() == []


def test_max_free_mb():
    gpus = [{"free_mb": 100}, {"free_mb": 500}, {"free_mb": 250}]
    assert gpu.max_free_mb(gpus) == 500


def test_max_free_mb_empty_is_none():
    assert gpu.max_free_mb([]) is None
