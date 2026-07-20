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


def test_run_detection_samples_the_unexplained_spill_not_the_raw_non_local(audit_spy):
    """A deliberately CPU-offloaded model shows up as its runner's Non Local
    Usage. Recording that as spill would make trend() report the GPU 'spilling
    to system RAM' for the whole life of a normal 32B load."""
    status = _status(gpus=[{"used_mb": 23768, "total_mb": 24576}])
    status["pressure"] = {"free_mb": 559, "non_local_mb": 3950,
                          "explained_offload_mb": 3906,
                          "unexplained_spill_mb": 44, "state": "degraded"}
    server._run_detection(status)
    sample = audit_spy["sample"][0]
    assert sample["spill_mb"] == 44
    # the conflated figure must not be persisted under its old name
    assert "non_local_mb" not in sample


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


def test_reserve_tool_surfaces_a_rejected_gb_as_a_structured_error(monkeypatch, tmp_path):
    """A bad argument must degrade like every other ledger failure — an
    {ok: False, summary} payload, never a raw traceback through MCP.

    The ledger path is redirected explicitly: this call reaches the REAL
    ``_claims.reserve``, and it writes nothing today only because validation
    happens to run before the lock is taken. Relying on that ordering would let
    a future reshuffle quietly start editing the user's live claims.json."""
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", tmp_path / "claims.json")
    result = server._reserve_impl(-8.0, "sneaky", "cancel yours", 60, None)
    assert result["ok"] is False
    assert "gb" in result["summary"]
    assert not (tmp_path / "claims.json").exists()


def test_trend_empty_window_names_missing_gpu_readings_as_a_cause(monkeypatch):
    """Sampling is skipped whenever a status call has no GPU rows — the case for
    anyone without a working nvidia-smi. Blaming only the throttle and
    VRAM_MCP_AUDIT=0 sends that user chasing two causes that aren't theirs."""
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: [])
    text = server._trend_impl(1.0)["summary"]
    assert "nvidia-smi" in text
    assert "VRAM_MCP_AUDIT=0" in text


# ---- warm() admission ------------------------------------------------------

@pytest.fixture
def warm_env(monkeypatch):
    """A GPU with 20 GB free and one 8 GB reservation held by another session."""
    monkeypatch.setattr(server._audit, "log_action", lambda **kwargs: None)
    monkeypatch.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": 20000, "pressure": {}})
    monkeypatch.setattr(server, "_active_claims", lambda: (
        [{"kind": "reservation", "gb": 8, "owner": "trainer", "purpose": "sd"}], True))
    monkeypatch.setattr(server._ollama, "warm", lambda model, keep_alive: True)
    return monkeypatch


def test_warm_allowed_with_unverifiable_size_says_the_check_was_unverified(warm_env):
    """tags() returns {} on ANY failure, so a timed-out /api/tags makes EVERY
    model 'size unknown' and admission control degrades wholesale to allow. That
    is the deliberate fail-open policy — but a 20 GB model must not sail through
    an 8 GB reservation reporting a plain success."""
    warm_env.setattr(server._ollama, "tags", lambda: {})
    result = server._warm_impl("qwen3:32b", "5m", "tester", False)
    assert result["ok"] is True
    assert result["reason"] == "size_unknown"
    assert result["size_verified"] is False
    assert "size" in result["summary"] and "8192 MB" in result["summary"]


def test_warm_allowed_with_a_verified_size_is_distinguishable(warm_env):
    warm_env.setattr(server._ollama, "tags", lambda: {"qwen3:32b": 1900})
    result = server._warm_impl("qwen3:32b", "5m", "tester", False)
    assert result["ok"] is True
    assert result["reason"] == "fits"
    assert result["size_verified"] is True
    assert "could not be verified" not in result["summary"]


def test_warm_forced_reports_no_admission_verdict(warm_env):
    """force=True skips the check entirely; claiming a size was verified (or
    wasn't) would describe a check that never ran."""
    warm_env.setattr(server._ollama, "tags", lambda: {})
    result = server._warm_impl("qwen3:32b", "5m", "tester", True)
    assert result["ok"] is True
    assert "size_verified" not in result
    assert "reason" not in result


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
