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
    import os
    path = tmp_path / "claims.json"
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with pytest.raises(TimeoutError):
            with claims._locked(path, timeout=0.2, poll=0.05):
                pass
    finally:
        os.close(fd)
        os.remove(lock_path)
