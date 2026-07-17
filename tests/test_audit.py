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


def _holder(key, target, kind, size_mb=None):
    return {"key": key, "target": target, "kind": kind, "size_mb": size_mb}


def test_meaningful_holders_all_models_plus_big_processes():
    loaded = [{"name": "qwen3:8b"}]
    procs = [{"pid": 100, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
             {"pid": 200, "size_mb": 40, "name": "chrome.exe", "cmdline": "chrome", "kind": "graphics"}]
    holders = audit.meaningful_holders(loaded, procs, threshold_mb=512)
    keys = {h["key"] for h in holders}
    assert "ollama:qwen3:8b" in keys
    assert "process:100" in keys        # 14.5 GB -> meaningful
    assert "process:200" not in keys    # 40 MB -> noise, filtered


def test_detect_first_run_sets_baseline_no_events(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    holders = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    emitted = audit.detect_and_log(holders, last_seen_path=ls, log_path=log, now_fn=_clock())
    assert emitted == []                       # no false "everything disappeared"
    assert audit.read_events(path=log) == []


def test_detect_disappearance_of_process_is_unattributed(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    p = [_holder("process:100", "python.exe train_lora_kg.py", "process", 14492)]
    audit.detect_and_log(p, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # gone
    assert len(emitted) == 1
    e = emitted[0]
    assert e["type"] == "disappeared" and e["kind"] == "process"
    assert e["cause"] == "unattributed" and e["size_mb"] == 14492
    assert "python.exe train_lora_kg.py" in e["target"]


def test_detect_ollama_disappearance_attributed_to_recent_unload(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(2)
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", now_fn=clk, path=log)             # our action
    clk.tick(1)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "self_action" and emitted[0]["actor"] == "retro-repo"


def test_detect_ollama_disappearance_without_action_is_external(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "external"


def test_detect_appearance_logged(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # empty baseline
    m = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
    emitted = audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["type"] == "appeared" and emitted[0]["target"] == "llama3.2"


def test_detect_second_session_does_not_double_log(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session A logs it
    again = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session B
    assert again == []  # already gone from last_seen -> nothing to re-log
