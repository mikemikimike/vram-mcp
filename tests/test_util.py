"""Tests for vram_mcp._util shared file/lock/json helpers."""
import json
import os
from datetime import datetime, timezone

import pytest

from vram_mcp import _util


def test_iso_roundtrip():
    dt = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)
    assert _util.iso(dt) == "2026-07-16T18:00:00Z"
    assert _util.parse_iso("2026-07-16T18:00:00Z") == dt


def test_save_and_load_json(tmp_path):
    p = tmp_path / "d.json"
    _util.save_json_atomic(p, {"k": [1, 2]})
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict)) == {"k": [1, 2]}


def test_load_json_missing_returns_default(tmp_path):
    p = tmp_path / "nope.json"
    assert _util.load_json(p, lambda: {"k": []}, lambda d: True) == {"k": []}


def test_load_json_corrupt_quarantines_and_defaults(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("not json {")
    out = _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict))
    assert out == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()  # preserved, not wiped


def test_load_json_wrong_shape_quarantines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([1, 2, 3]))  # valid JSON, wrong shape
    out = _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict) and "k" in d)
    assert out == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()


def test_save_json_retries_replace_on_permission_error(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(_util.os, "replace", flaky)
    _util.save_json_atomic(tmp_path / "d.json", {"ok": True})
    assert calls["n"] == 4


def test_locked_breaks_stale_lock(tmp_path):
    import time
    p = tmp_path / "d.json"
    lock = p.with_suffix(p.suffix + ".lock")
    lock.write_text("9999 old\n")
    old = time.time() - (_util.LOCK_STALE_SECONDS + 30)
    os.utime(lock, (old, old))
    with _util.locked(p, timeout=1.0):
        pass  # acquired by breaking the stale lock
    assert not lock.exists()


def test_locked_times_out_on_live_lock(tmp_path):
    p = tmp_path / "d.json"
    lock = p.with_suffix(p.suffix + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.utime(lock, None)  # fresh -> live
    try:
        with pytest.raises(TimeoutError):
            with _util.locked(p, timeout=0.2, poll=0.05):
                pass
    finally:
        os.close(fd)
        os.remove(lock)


def test_append_jsonl_capped_prunes_to_last_n(tmp_path):
    p = tmp_path / "e.jsonl"
    for i in range(10):
        _util.append_jsonl_capped(p, {"i": i}, cap=5)
    rows = _util.read_jsonl(p)
    assert [r["i"] for r in rows] == [5, 6, 7, 8, 9]


def test_read_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a":1}\nGARBAGE\n{"a":2}\n')
    assert _util.read_jsonl(p) == [{"a": 1}, {"a": 2}]
