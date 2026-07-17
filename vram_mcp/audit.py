"""Append-only, cause-attributed VRAM audit log + disappearance detector.

One typed ``events.jsonl`` (``type`` in action|disappeared|appeared) plus a
``last_seen.json`` diff baseline. Bounded (``cap`` events, pruned on append),
crash-safe (atomic writes + the shared file lock). The audit MUST NEVER break a
tool call: every write is best-effort. Pure w.r.t. time (``now_fn`` injected)
and paths (injected in tests).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ._util import append_jsonl_capped, iso, load_json, locked, parse_iso, read_jsonl, save_json_atomic

DEFAULT_EVENTS_PATH = Path.home() / ".cache" / "vram-mcp" / "events.jsonl"
DEFAULT_LAST_SEEN_PATH = Path.home() / ".cache" / "vram-mcp" / "last_seen.json"

_RECENT_ACTION_SECONDS = 10.0


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def log_action(*, action: str, target: str, kind: str, actor: str = "unknown",
               force: bool = False, outcome: str = "ok", detail: str = "",
               now_fn=_default_now, path: Path = DEFAULT_EVENTS_PATH,
               cap: int = 5000) -> None:
    """Append one ``type="action"`` event. Best-effort — never raises."""
    try:
        event = {
            "ts": iso(now_fn()), "type": "action", "kind": kind, "target": target,
            "action": action, "actor": actor, "force": force, "outcome": outcome,
            "detail": detail,
        }
        with locked(path):
            append_jsonl_capped(path, event, cap)
    except Exception:
        pass  # the audit may never break a tool call


def read_events(*, model: str | None = None, type: str | None = None,
                limit: int = 50, since: str | None = None,
                path: Path = DEFAULT_EVENTS_PATH) -> list[dict]:
    """Recent events, newest-first, optionally filtered by target/type/since."""
    rows = read_jsonl(path)
    if model is not None:
        rows = [r for r in rows if r.get("target") == model]
    if type is not None:
        rows = [r for r in rows if r.get("type") == type]
    if since is not None:
        rows = [r for r in rows if r.get("ts", "") >= since]  # ISO-Z sorts lexically
    rows.reverse()
    return rows[:limit]


def meaningful_holders(loaded_models, process_table, threshold_mb: int) -> list[dict]:
    """Holders worth tracking: every Ollama model, plus non-Ollama processes
    holding >= ``threshold_mb`` dedicated VRAM (size is the filter, not names)."""
    holders = []
    for m in loaded_models:
        name = m.get("name")
        if not name:
            continue
        holders.append({"key": f"ollama:{name}", "target": name,
                        "kind": "ollama", "size_mb": m.get("size_vram_mb")})
    for p in process_table:
        size = p.get("size_mb")
        if size is not None and size >= threshold_mb:
            label = p.get("cmdline") or p.get("name") or f"pid {p['pid']}"
            holders.append({"key": f"process:{p['pid']}", "target": label,
                            "kind": "process", "size_mb": size})
    return holders


def _recent_action_for(target: str, now: datetime, log_path: Path):
    """The most recent unload/ensure_free action on ``target`` within the
    attribution window, else None. ``read_events`` is newest-first."""
    floor = now.timestamp() - _RECENT_ACTION_SECONDS
    for e in read_events(model=target, type="action", limit=50, path=log_path):
        if e.get("action") not in ("unload", "ensure_free"):
            continue
        ts = e.get("ts")
        if not ts:
            continue
        try:
            when = parse_iso(ts).timestamp()
        except ValueError:
            continue
        if when >= floor:
            return e
    return None


def _disappeared_event(holder: dict, now: datetime, log_path: Path) -> dict:
    kind = holder["kind"]
    if kind == "ollama":
        action = _recent_action_for(holder["target"], now, log_path)
        if action is not None:
            return {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                    "target": holder["target"], "size_mb": holder.get("size_mb"),
                    "cause": "self_action", "actor": action.get("actor", "unknown"),
                    "detail": f"removed by {action.get('actor','unknown')} via {action.get('action')}"}
        return {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                "target": holder["target"], "size_mb": holder.get("size_mb"),
                "cause": "external",
                "detail": "no vram-mcp action recorded — Ollama idle-expiry, "
                          "memory-pressure eviction, or an external unload."}
    return {"ts": iso(now), "type": "disappeared", "kind": "process",
            "target": holder["target"], "size_mb": holder.get("size_mb"),
            "cause": "unattributed",
            "detail": "process exited or was killed; vram-mcp cannot observe the cause."}


def _valid_baseline_exists(path: Path) -> bool:
    """True only if ``path`` holds a readable, correctly-shaped baseline.
    Missing, unreadable, or wrong-shape all count as 'no baseline' -> a fresh
    first run that must NOT emit appeared/disappeared events."""
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("holders"), dict)


def detect_and_log(current_holders, *, last_seen_path: Path = DEFAULT_LAST_SEEN_PATH,
                   log_path: Path = DEFAULT_EVENTS_PATH, now_fn=_default_now,
                   cap: int = 5000) -> list[dict]:
    """Diff ``current_holders`` against last-seen under the lock; append
    disappeared/appeared events with attribution; rewrite last-seen. Returns the
    events emitted. Best-effort — never raises out to the caller."""
    emitted: list[dict] = []
    try:
        now = now_fn()
        with locked(last_seen_path):
            first_run = not _valid_baseline_exists(last_seen_path)
            prev = load_json(last_seen_path, lambda: {"holders": {}},
                             lambda d: isinstance(d, dict) and isinstance(d.get("holders"), dict))
            prev_map = prev["holders"]
            cur_map = {h["key"]: h for h in current_holders}

            for key, h in prev_map.items():
                if key not in cur_map:
                    emitted.append(_disappeared_event(h, now, log_path))
            if not first_run:
                for key, h in cur_map.items():
                    if key not in prev_map:
                        emitted.append({
                            "ts": iso(now), "type": "appeared", "kind": h["kind"],
                            "target": h["target"], "size_mb": h.get("size_mb"),
                            "detail": "now holding VRAM",
                        })

            for e in emitted:
                append_jsonl_capped(log_path, e, cap)
            save_json_atomic(last_seen_path, {"holders": cur_map})
    except Exception:
        return emitted
    return emitted
