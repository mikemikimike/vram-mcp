"""Append-only, cause-attributed VRAM audit log + disappearance detector.

One typed ``events.jsonl`` (``type`` in action|disappeared|appeared) plus a
``last_seen.json`` diff baseline. Bounded (``cap`` events, pruned on append),
crash-safe (atomic writes + the shared file lock). The audit MUST NEVER break a
tool call: every write is best-effort. Pure w.r.t. time (``now_fn`` injected)
and paths (injected in tests).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ._util import append_jsonl_capped, iso, locked, read_jsonl

DEFAULT_EVENTS_PATH = Path.home() / ".cache" / "vram-mcp" / "events.jsonl"
DEFAULT_LAST_SEEN_PATH = Path.home() / ".cache" / "vram-mcp" / "last_seen.json"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def log_action(*, action: str, target: str, kind: str, actor: str = "unknown",
               force: bool = False, outcome: str = "ok", detail: str = "",
               now_fn=_default_now, path: Path = DEFAULT_EVENTS_PATH,
               cap: int = 5000) -> None:
    """Append one ``type="action"`` event. Best-effort — never raises."""
    event = {
        "ts": iso(now_fn()), "type": "action", "kind": kind, "target": target,
        "action": action, "actor": actor, "force": force, "outcome": outcome,
        "detail": detail,
    }
    try:
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
