"""Tests for the thin wiring in vram_mcp.server.

The server module is the only one that imports ``mcp``; it is skipped rather
than failed where that package is absent, so the pure-module suite still runs
anywhere. Every audit call here is monkeypatched: these tests must never touch
the real ``~/.cache/vram-mcp`` files.
"""
import pytest

pytest.importorskip("mcp")

from vram_mcp import core, server  # noqa: E402


@pytest.fixture
def audit_spy(monkeypatch):
    """Record what _run_detection asks the audit to do, writing nothing."""
    calls = {"detect": [], "sample": []}

    def fake_detect(holders, **kwargs):
        calls["detect"].append(holders)
        return []

    def fake_sample(sample, **kwargs):
        calls["sample"].append(sample)
        return True

    monkeypatch.setattr(server._audit, "detect_and_log", fake_detect)
    monkeypatch.setattr(server._audit, "maybe_log_sample", fake_sample)
    return calls


def _status(gpus, loaded=(), procs=()):
    return {
        "gpus": list(gpus), "loaded": list(loaded),
        "other_processes": list(procs),
        "pressure": {"free_mb": None if not gpus else 500,
                     "non_local_mb": 0, "state": "ok"},
    }


def test_run_detection_skips_sample_when_gpu_data_absent(audit_spy):
    """list_loaded() deliberately skips the nvidia-smi spawn, so its status has
    no GPU rows. Recording {free_mb: None, used_mb: 0, total_mb: 0} would write
    a fiction AND burn the shared throttle slot, dropping the next REAL sample.
    Detection still has to run — only the sample is skipped."""
    server._run_detection(_status(gpus=[], loaded=[{"name": "m", "size_vram_mb": 8000}]))
    assert audit_spy["sample"] == []      # no fake zeros, no throttle slot burned
    assert len(audit_spy["detect"]) == 1  # holder diff still ran


def test_run_detection_samples_when_gpu_data_present(audit_spy):
    server._run_detection(_status(
        gpus=[{"used_mb": 20000, "total_mb": 24576}], loaded=[]))
    assert len(audit_spy["sample"]) == 1
    assert audit_spy["sample"][0]["total_mb"] == 24576


def test_full_status_forwards_the_spill_threshold(monkeypatch):
    """VRAM_MCP_SPILL_MB is only a knob if it actually reaches pressure()."""
    seen = {}

    def fake_combined_status(gpu_fn, ollama, **kwargs):
        seen.update(kwargs)
        return {"gpus": [], "loaded": [], "free_mb": None, "pressure": {}}

    monkeypatch.setattr(core, "combined_status", fake_combined_status)
    monkeypatch.setattr(server, "_SPILL_MB", 777)
    server._full_status()
    assert seen["spill_threshold_mb"] == 777


def test_reserve_tool_surfaces_a_rejected_gb_as_a_structured_error():
    """A bad argument must degrade like every other ledger failure — an
    {ok: False, summary} payload, never a raw traceback through MCP."""
    result = server._reserve_impl(-8.0, "sneaky", "cancel yours", 60, None)
    assert result["ok"] is False
    assert "gb" in result["summary"]


def test_trend_caps_returned_samples(monkeypatch):
    """trend() output lands in an agent's context window; 10k raw rows would
    flood it. The summary still covers every row."""
    rows = [{"ts": f"2026-07-19T00:00:{i % 60:02d}Z", "type": "sample",
             "free_mb": 1000 + i, "state": "ok"} for i in range(500)]
    # read_events is newest-first; _trend_impl reverses to oldest-first.
    monkeypatch.setattr(server._audit, "read_events",
                        lambda **kwargs: list(reversed(rows)))
    result = server._trend_impl(1.0)
    assert result["count"] == 500                 # summary saw everything
    assert len(result["samples"]) == server._TREND_SAMPLE_CAP
    assert result["samples_truncated"] is True
    assert result["samples"][-1]["free_mb"] == 1499   # the newest rows kept
    assert str(server._TREND_SAMPLE_CAP) in result["summary"]


def test_trend_does_not_flag_truncation_when_everything_fits(monkeypatch):
    rows = [{"ts": "2026-07-19T00:00:00Z", "type": "sample",
             "free_mb": 1000, "state": "ok"}]
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: rows)
    result = server._trend_impl(1.0)
    assert result["samples_truncated"] is False
    assert len(result["samples"]) == 1


def test_trend_summary_says_unknown_when_the_newest_sample_has_no_value(monkeypatch):
    rows = [{"ts": "2026-07-19T00:00:01Z", "type": "sample", "free_mb": None,
             "state": "ok"},
            {"ts": "2026-07-19T00:00:00Z", "type": "sample", "free_mb": 900,
             "state": "ok"}]  # newest-first, as read_events returns
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: rows)
    result = server._trend_impl(1.0)
    assert result["latest_free_mb"] is None
    assert "now unknown" in result["summary"]
