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

# A lock file older than this is presumed abandoned (its holder was killed
# without running the ``finally`` cleanup) and is broken by the next waiter.
# Real critical sections here are milliseconds long, so 30s is very generous.
_LOCK_STALE_SECONDS = 30.0

# Windows: a concurrent lock-free reader that momentarily has claims.json
# open makes ``os.replace`` fail with a sharing violation (PermissionError,
# WinError 5) because CPython's ``open`` doesn't pass FILE_SHARE_DELETE.
# Readers close the file quickly, so a short retry loop rides it out.
_REPLACE_ATTEMPTS = 10
_REPLACE_RETRY_SLEEP = 0.02


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
    honor ``O_EXCL``). Raises ``TimeoutError`` if a LIVE lock can't be
    acquired within ``timeout`` seconds (another process holds it).

    A lock whose mtime is older than ``_LOCK_STALE_SECONDS`` is presumed
    abandoned (holder hard-killed before its cleanup ran) and is removed so
    one dead process can't brick every future session. The holder's PID and
    an ISO timestamp are written into the lock file purely as forensic info.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock_path).st_mtime
            except OSError:
                # Lock vanished between open() and stat() — retry right away.
                continue
            if age > _LOCK_STALE_SECONDS:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass  # another waiter broke it first — fine
                continue  # retry immediately, no poll sleep
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(poll)
    try:
        os.write(fd, f"{os.getpid()} {_iso(_default_now())}\n".encode("utf-8"))
    except OSError:
        pass  # forensic info only — never fail acquisition over it
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _quarantine(path: Path) -> None:
    """Move an unreadable/wrong-shape ledger aside (best-effort) so the next
    write doesn't silently destroy it."""
    try:
        os.replace(path, path.with_suffix(path.suffix + ".corrupt"))
    except OSError:
        pass


def _load(path: Path) -> dict:
    if not path.exists():
        return {"claims": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict) or not isinstance(doc.get("claims"), list):
        # File exists but is corrupt or the wrong shape: preserve it as
        # claims.json.corrupt instead of letting the next _save wipe it.
        _quarantine(path)
        return {"claims": []}
    return doc


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows sharing violation: a lock-free reader briefly has the
            # destination open. Retry; readers finish in well under 200ms.
            if attempt == _REPLACE_ATTEMPTS - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(_REPLACE_RETRY_SLEEP)


def _is_active(record: dict, now: datetime) -> bool:
    try:
        return now < _parse_iso(record["expires_at"])
    except (KeyError, ValueError, TypeError):
        # Malformed record (missing/unparsable expires_at, naive datetime,
        # non-dict entry, ...): treat as expired rather than crash every tool.
        return False


def _prune_expired(data: dict, now: datetime) -> None:
    """Drop expired/malformed records in place (called under the lock, right
    before a write that's happening anyway — keeps the file bounded)."""
    data["claims"] = [r for r in data["claims"] if _is_active(r, now)]


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
        _prune_expired(data, now)
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}


def renew(
    claim_id: str, ttl_seconds: Optional[int] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Extend a claim's expiry to ``now + ttl_seconds`` (or its original TTL).

    Returns ``{"ok": False}`` if no claim with ``claim_id`` exists or the
    claim has already expired — an expired claim is not resurrected.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    with _locked(path):
        data = _load(path)
        for record in data["claims"]:
            if isinstance(record, dict) and record.get("claim_id") == claim_id:
                if not _is_active(record, now):
                    return {"ok": False}
                ttl = ttl_seconds if ttl_seconds is not None else record["ttl_seconds"]
                record["ttl_seconds"] = ttl
                record["renewed_at"] = _iso(now)
                record["expires_at"] = _iso(now + timedelta(seconds=ttl))
                _prune_expired(data, now)  # renewed record is active, so kept
                _save(path, data)
                return {"ok": True, "expires_at": record["expires_at"]}
    return {"ok": False}


def release(claim_id: str, *, path: Optional[Path] = None) -> dict:
    """Remove a claim immediately (before its TTL expires)."""
    path = path or _DEFAULT_PATH
    with _locked(path):
        data = _load(path)
        before = len(data["claims"])
        data["claims"] = [
            r for r in data["claims"]
            if not (isinstance(r, dict) and r.get("claim_id") == claim_id)
        ]
        found = len(data["claims"]) != before
        if found:
            _prune_expired(data, _default_now())
            _save(path, data)
    return {"ok": found}


def list_claims(
    model: Optional[str] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> list[dict]:
    """Every currently-ACTIVE claim, optionally filtered to one model.

    Expired claims are silently excluded here and physically pruned by the
    next locked write (``claim()``/``renew()``/``release()`` all write).
    This read is lock-free: writes land via atomic ``os.replace``, so a
    reader always sees a complete file (old or new) — never a torn one. The
    flip side on Windows is that a reader holding the file open can make a
    concurrent writer's ``os.replace`` fail with a sharing violation, which
    ``_save`` handles by retrying briefly.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    active = [r for r in _load(path)["claims"] if _is_active(r, now)]
    if model is not None:
        active = [r for r in active if r["model"] == model]
    return active
