"""Shared, file-based attribution ledger.

Every ``vram-mcp`` session runs its own independent subprocess (no shared
server), so cross-session attribution needs a shared file on disk:
``~/.cache/vram-mcp/claims.json`` by default. Writes are atomic (temp file +
``os.replace``) and serialized with a sibling lock file so concurrent
sessions can't corrupt each other's writes. A claim expires via TTL — no
explicit release is required, so a session that crashes without cleanup
doesn't leave a permanently-stuck claim.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

_DEFAULT_PATH = Path.home() / ".cache" / "vram-mcp" / "claims.json"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@contextmanager
def _locked(path: Path, timeout: float = 5.0, poll: float = 0.05):
    """Serialize access to ``path`` via a sibling ``.lock`` file.

    Uses ``os.open`` with ``O_CREAT | O_EXCL`` — atomic file creation that
    fails if the lock already exists, cross-platform (Windows and POSIX both
    honor ``O_EXCL``). Raises ``TimeoutError`` if the lock can't be acquired
    within ``timeout`` seconds (another process holds it).
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _load(path: Path) -> dict:
    if not path.exists():
        return {"claims": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"claims": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _is_active(record: dict, now: datetime) -> bool:
    return now < _parse_iso(record["expires_at"])


def claim(
    model: str, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Register that ``owner`` is using ``model`` for ``purpose``.

    Returns ``{"claim_id", "expires_at"}``. Multiple claims may exist for the
    same model at once (different owners); each is independent.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    expires_at = now + timedelta(seconds=ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex,
        "model": model, "owner": owner, "purpose": purpose,
        "claimed_at": _iso(now), "renewed_at": _iso(now),
        "ttl_seconds": ttl_seconds, "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}


def renew(
    claim_id: str, ttl_seconds: Optional[int] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Extend a claim's expiry to ``now + ttl_seconds`` (or its original TTL).

    Returns ``{"ok": False}`` if no claim with ``claim_id`` exists.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    with _locked(path):
        data = _load(path)
        for record in data["claims"]:
            if record["claim_id"] == claim_id:
                ttl = ttl_seconds if ttl_seconds is not None else record["ttl_seconds"]
                record["ttl_seconds"] = ttl
                record["renewed_at"] = _iso(now)
                record["expires_at"] = _iso(now + timedelta(seconds=ttl))
                _save(path, data)
                return {"ok": True, "expires_at": record["expires_at"]}
    return {"ok": False}


def release(claim_id: str, *, path: Optional[Path] = None) -> dict:
    """Remove a claim immediately (before its TTL expires)."""
    path = path or _DEFAULT_PATH
    with _locked(path):
        data = _load(path)
        before = len(data["claims"])
        data["claims"] = [r for r in data["claims"] if r["claim_id"] != claim_id]
        found = len(data["claims"]) != before
        if found:
            _save(path, data)
    return {"ok": found}


def list_claims(
    model: Optional[str] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> list[dict]:
    """Every currently-ACTIVE claim, optionally filtered to one model.

    Expired claims are silently excluded (not deleted — ``claim()``/
    ``release()`` are the only writers; a read never needs the lock since
    writes are atomic, so a reader always sees a complete file, old or new).
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    active = [r for r in _load(path)["claims"] if _is_active(r, now)]
    if model is not None:
        active = [r for r in active if r["model"] == model]
    return active
