"""Tests for vram_mcp.audit — event log + diff detector, real tmp_path files."""
from datetime import datetime, timedelta, timezone

from vram_mcp import audit

_T0 = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)


def _clock(start=_T0):
    st = {"now": start}
    def now_fn():
        return st["now"]
    def tick(s):
        st["now"] = st["now"] + timedelta(seconds=s)
    now_fn.tick = tick
    return now_fn


def test_log_action_writes_typed_event(tmp_path):
    p = tmp_path / "events.jsonl"
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", force=True, outcome="ok",
                     detail="unloaded", now_fn=_clock(), path=p)
    events = audit.read_events(path=p)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "action" and e["action"] == "unload"
    assert e["target"] == "qwen3:8b" and e["actor"] == "retro-repo"
    assert e["force"] is True and e["ts"] == "2026-07-16T18:00:00Z"


def test_read_events_newest_first_and_limit(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for name in ["a", "b", "c"]:
        audit.log_action(action="warm", target=name, kind="ollama",
                         now_fn=clk, path=p)
        clk.tick(1)
    got = audit.read_events(limit=2, path=p)
    assert [e["target"] for e in got] == ["c", "b"]


def test_read_events_filter_by_model_and_type(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama", now_fn=clk, path=p)
    audit.log_action(action="warm", target="llama3.2", kind="ollama", now_fn=clk, path=p)
    assert [e["target"] for e in audit.read_events(model="qwen3:8b", path=p)] == ["qwen3:8b"]
    assert [e["action"] for e in audit.read_events(type="action", path=p)]  # all are actions


def test_read_events_since_floor(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="warm", target="old", kind="ollama", now_fn=clk, path=p)
    clk.tick(120)
    audit.log_action(action="warm", target="new", kind="ollama", now_fn=clk, path=p)
    got = audit.read_events(since="2026-07-16T18:01:00Z", path=p)
    assert [e["target"] for e in got] == ["new"]


def test_log_action_respects_cap(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for i in range(8):
        audit.log_action(action="warm", target=str(i), kind="ollama",
                         now_fn=clk, path=p, cap=5)
    targets = [e["target"] for e in audit.read_events(limit=99, path=p)]
    assert targets == ["7", "6", "5", "4", "3"]  # newest-first, last 5
