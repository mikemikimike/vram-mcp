"""Tests for vram_mcp._util shared file/lock/json helpers."""
import json
import os
import subprocess
import sys
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
    assert _util.load_json(tmp_path / "nope.json", lambda: {"k": []}, lambda d: True) == {"k": []}


def test_load_json_corrupt_quarantines_and_defaults(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("not json {")
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict)) == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()


def test_load_json_wrong_shape_quarantines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict) and "k" in d) == {"k": []}
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


def _holder_script(path, mode):
    body = "    os._exit(0)" if mode == "crash" else "    import time; time.sleep(2)"
    return (
        "import os, sys\nfrom pathlib import Path\nfrom vram_mcp._util import locked\n"
        "p = Path(sys.argv[1])\nwith locked(p):\n"
        "    os.utime(p.with_suffix(p.suffix + '.lock'), (1, 1))\n"
        "    print('locked', flush=True)\n" + body + "\n"
    )


def test_locked_live_aged_holder_remains_exclusive(tmp_path):
    p = tmp_path / "d.json"
    proc = subprocess.Popen([sys.executable, "-c", _holder_script(p, "hold"), str(p)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(TimeoutError):
            with _util.locked(p, timeout=0.1, poll=0.01):
                pass
    finally:
        proc.wait(timeout=5)
    assert p.with_suffix(p.suffix + ".lock").exists()


def test_locked_is_released_when_holder_crashes(tmp_path):
    p = tmp_path / "d.json"
    proc = subprocess.Popen([sys.executable, "-c", _holder_script(p, "crash"), str(p)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    assert proc.wait(timeout=5) == 0
    with _util.locked(p, timeout=0.5):
        pass


def test_append_jsonl_capped_prunes_to_last_n(tmp_path):
    p = tmp_path / "e.jsonl"
    for i in range(10):
        _util.append_jsonl_capped(p, {"i": i}, cap=5)
    assert [r["i"] for r in _util.read_jsonl(p)] == [5, 6, 7, 8, 9]


def test_read_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a":1}\nGARBAGE\n{"a":2}\n')
    assert _util.read_jsonl(p) == [{"a": 1}, {"a": 2}]
