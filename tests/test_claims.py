"""Tests for vram_mcp.claims — real temp-dir file I/O (atomicity is the point)."""

from datetime import datetime, timedelta, timezone

import pytest

from vram_mcp import claims

# Shared test epoch: every test pins the clock here (in the past relative to
# wall time — which is exactly why write paths must honor now_fn, never the
# real clock, or wall-clock pruning would eat these fixtures).
_T0 = datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc)


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
    now_fn = _clock(_T0)
    result = claims.claim("llama3.2", "project-a", "narration",
                          ttl_seconds=3600, path=path, now_fn=now_fn)
    assert "claim_id" in result
    assert result["expires_at"] == "2026-07-13T19:00:00Z"

    active = claims.list_claims(path=path, now_fn=now_fn)
    assert len(active) == 1
    assert active[0]["model"] == "llama3.2"
    assert active[0]["owner"] == "project-a"
    assert active[0]["purpose"] == "narration"


def test_list_claims_filters_by_model(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    claims.claim("qwen3:8b", "b", "y", path=path, now_fn=now_fn)

    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 2
    only_qwen = claims.list_claims("qwen3:8b", path=path, now_fn=now_fn)
    assert len(only_qwen) == 1
    assert only_qwen[0]["owner"] == "b"


def test_claim_expires_after_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(30)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1

    now_fn.tick(31)  # 61s total -> past the 60s ttl
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_renew_extends_expiry_with_original_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(50)
    result = claims.renew(created["claim_id"], path=path, now_fn=now_fn)
    assert result["ok"] is True
    assert result["expires_at"] == "2026-07-13T18:01:50Z"  # now(18:00:50) + 60s

    now_fn.tick(55)  # 105s since creation, 55s since renew -> still active
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1


def test_renew_with_new_ttl_overrides(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    result = claims.renew(created["claim_id"], ttl_seconds=7200, path=path, now_fn=now_fn)
    assert result["expires_at"] == "2026-07-13T20:00:00Z"


def test_renew_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.renew("nonexistent-id", path=path) == {"ok": False}


def test_release_removes_claim(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert claims.release(created["claim_id"], path=path) == {"ok": True}
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_release_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.release("nonexistent", path=path) == {"ok": False}


def test_multiple_claims_same_model_independent(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "session-a", "reason-a", path=path, now_fn=now_fn)
    claims.claim("llama3.2", "session-b", "reason-b", path=path, now_fn=now_fn)
    active = claims.list_claims("llama3.2", path=path, now_fn=now_fn)
    assert {c["owner"] for c in active} == {"session-a", "session-b"}


def test_sequential_claims_both_persist(tmp_path):
    """Each claim()/release() call round-trips through the lock cleanly --
    a stale lock from a prior call never blocks the next one."""
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
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
        os.utime(lock_path, None)  # mtime = NOW: a LIVE lock, not a stale one
        with pytest.raises(TimeoutError):
            with claims._locked(path, timeout=0.2, poll=0.05):
                pass
    finally:
        os.close(fd)
        os.remove(lock_path)


# --- FIX 1: stale locks from hard-killed holders are broken, live ones honored


def test_stale_lock_is_broken_and_claim_succeeds(tmp_path):
    import os
    import time
    path = tmp_path / "claims.json"
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.write_text("9999 2026-07-13T00:00:00Z\n")  # abandoned holder
    old = time.time() - (claims._LOCK_STALE_SECONDS + 30)
    os.utime(lock_path, (old, old))

    now_fn = _clock(_T0)
    result = claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert "claim_id" in result
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1
    assert not lock_path.exists()  # broken, then released after the write


def test_stale_lock_break_tolerates_losing_the_removal_race(tmp_path):
    """os.remove of the stale lock may hit FileNotFoundError if another
    process broke it first — _locked must swallow that and retry."""
    import os
    import time
    path = tmp_path / "claims.json"
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.write_text("stale\n")
    old = time.time() - (claims._LOCK_STALE_SECONDS + 30)
    os.utime(lock_path, (old, old))

    real_remove = os.remove

    def racing_remove(p):
        real_remove(p)  # "another process" already removed it...
        raise FileNotFoundError(2, "gone", str(p))  # ...so ours would raise

    raced = {"done": False}

    def remove_once(p):
        if not raced["done"] and str(p) == str(lock_path):
            raced["done"] = True
            return racing_remove(p)
        return real_remove(p)

    original = claims.os.remove
    claims.os.remove = remove_once
    try:
        with claims._locked(path, timeout=1.0):
            pass
    finally:
        claims.os.remove = original
    assert raced["done"]


# --- FIX 2: _save retries os.replace on Windows sharing violations


def test_save_retries_replace_on_permission_error_then_succeeds(tmp_path, monkeypatch):
    import os
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(claims.os, "replace", flaky_replace)
    now_fn = _clock(_T0)
    result = claims.claim("llama3.2", "a", "x", path=tmp_path / "claims.json",
                          now_fn=now_fn)
    assert "claim_id" in result
    assert calls["n"] == 4
    active = claims.list_claims(path=tmp_path / "claims.json", now_fn=now_fn)
    assert len(active) == 1


def test_save_gives_up_after_retries_and_cleans_tmp(tmp_path, monkeypatch):
    path = tmp_path / "claims.json"

    def always_fails(src, dst):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(claims.os, "replace", always_fails)
    now_fn = _clock(_T0)
    with pytest.raises(PermissionError):
        claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert not path.with_suffix(path.suffix + ".tmp").exists()


# --- FIX 3: malformed records are tolerated, not fatal


def test_malformed_records_are_skipped_not_fatal(tmp_path):
    import json
    path = tmp_path / "claims.json"
    good = {
        "claim_id": "good", "model": "llama3.2", "owner": "a", "purpose": "x",
        "claimed_at": "2026-07-13T18:00:00Z", "renewed_at": "2026-07-13T18:00:00Z",
        "ttl_seconds": 3600, "expires_at": "2026-07-13T19:00:00Z",
    }
    bad_missing_key = {"claim_id": "b1", "model": "m"}  # no expires_at
    bad_unparsable = {"claim_id": "b2", "expires_at": "not-a-date"}
    bad_naive_dt = {"claim_id": "b3", "expires_at": "2026-07-13T19:00:00"}  # no tz
    bad_not_a_dict = "garbage"
    path.write_text(json.dumps({"claims": [
        good, bad_missing_key, bad_unparsable, bad_naive_dt, bad_not_a_dict,
    ]}), encoding="utf-8")

    now_fn = _clock(_T0)
    active = claims.list_claims(path=path, now_fn=now_fn)  # must not raise
    assert [r["claim_id"] for r in active] == ["good"]

    # Writes also survive: malformed records get pruned, good one kept.
    claims.claim("qwen3:8b", "b", "y", path=path, now_fn=now_fn)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk["claims"]) == 2
    assert {r["model"] for r in on_disk["claims"]} == {"llama3.2", "qwen3:8b"}


# --- FIX 4: corrupt / wrong-shape files are preserved aside, not wiped


@pytest.mark.parametrize("content", ["[]", "{}", "{{{not json", '{"claims": 42}'])
def test_corrupt_file_renamed_aside_and_ops_continue(tmp_path, content):
    path = tmp_path / "claims.json"
    corrupt_path = path.with_suffix(path.suffix + ".corrupt")
    path.write_text(content, encoding="utf-8")

    now_fn = _clock(_T0)
    assert claims.list_claims(path=path, now_fn=now_fn) == []  # no raise
    assert corrupt_path.exists()
    assert corrupt_path.read_text(encoding="utf-8") == content  # preserved

    # Subsequent operations work on a fresh ledger.
    created = claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1
    assert claims.release(created["claim_id"], path=path) == {"ok": True}


def test_missing_file_is_not_treated_as_corrupt(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    assert claims.list_claims(path=path, now_fn=now_fn) == []
    assert not path.with_suffix(path.suffix + ".corrupt").exists()


# --- FIX 5: renew doesn't resurrect expired claims; writes prune the file


def test_renew_of_expired_claim_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60,
                           path=path, now_fn=now_fn)
    now_fn.tick(61)  # past the TTL
    assert claims.renew(created["claim_id"], path=path, now_fn=now_fn) == {"ok": False}
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_expired_records_are_pruned_by_next_write(tmp_path):
    import json
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("old-model", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    now_fn.tick(61)  # first claim expires
    claims.claim("new-model", "b", "y", ttl_seconds=60, path=path, now_fn=now_fn)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk["claims"]) == 1  # expired record physically removed
    assert on_disk["claims"][0]["model"] == "new-model"


def test_renew_prunes_expired_records_but_keeps_renewed_one(tmp_path):
    import json
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("doomed", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    keeper = claims.claim("keeper", "b", "y", ttl_seconds=7200,
                          path=path, now_fn=now_fn)
    now_fn.tick(61)  # "doomed" expires, "keeper" still active
    result = claims.renew(keeper["claim_id"], path=path, now_fn=now_fn)
    assert result["ok"] is True

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert [r["model"] for r in on_disk["claims"]] == ["keeper"]


def test_release_prunes_with_injected_clock_sibling_survives(tmp_path):
    """release() must prune with now_fn, not the wall clock: the test epoch is
    in the past relative to real time, so wall-clock pruning would silently
    destroy the surviving sibling claim."""
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    keeper = claims.claim("llama3.2", "keeper", "still-working",
                          ttl_seconds=3600, path=path, now_fn=now_fn)
    goner = claims.claim("qwen3:8b", "goner", "done", path=path, now_fn=now_fn)

    assert claims.release(goner["claim_id"], path=path, now_fn=now_fn) == {"ok": True}

    survivors = claims.list_claims(path=path, now_fn=now_fn)
    assert [c["claim_id"] for c in survivors] == [keeper["claim_id"]]
