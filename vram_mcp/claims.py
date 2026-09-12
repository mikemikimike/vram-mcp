"""Durable, cross-session claims, reservations, and pending operations.

The JSON ledger is guarded only while it is read or changed. Long-running
backend work uses a durable operation lease instead of holding a file lock
across HTTP. A crashed caller can delay a same-model operation only until its
lease expires; it can never leave the ledger permanently locked.
"""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ._util import (
    locked as _locked, iso as _iso, parse_iso as _parse_iso,
    save_json_atomic, _try_lock, _unlock,
)
from .models import canonical_model
from .validation import nonblank_text, positive_gb, positive_ttl_seconds

_DEFAULT_PATH = Path.home() / ".cache" / "vram-mcp" / "claims.json"
_OPERATION_LEASE_SECONDS = 120
_EVICTION_KINDS = {"unload", "ensure_free"}
_DEFAULT_SCOPE = "gpu:index=0"
_OPERATION_LOCKS: dict[str, int] = {}


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(now: datetime, ttl_seconds: int) -> datetime:
    try:
        return now + timedelta(seconds=ttl_seconds)
    except OverflowError as exc:
        raise ValueError("ttl_seconds exceeds the supported datetime range") from exc


def _operation_scope(record: dict) -> str:
    """Read legacy no-scope operation records as the original default GPU."""
    return nonblank_text(record.get("scope", _DEFAULT_SCOPE), "scope")


def _load(path: Path) -> dict:
    """Read strictly: a missing ledger is empty; an existing bad one is unsafe."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"claims": [], "operations": []}
    except OSError:
        raise
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"claim ledger is corrupt: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError(f"claim ledger has invalid shape: {path}")
    operations = data.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError(f"claim ledger has invalid operations: {path}")
    for record in operations:
        try:
            if not isinstance(record, dict):
                raise ValueError
            nonblank_text(record.get("operation_id"), "operation_id")
            canonical_model(record.get("model"))
            nonblank_text(record.get("kind"), "kind")
            _operation_scope(record)
            expires_at = _parse_iso(record["expires_at"])
            if expires_at.tzinfo is None:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"claim ledger has malformed operation: {path}")
    data.setdefault("operations", operations)
    return data


def _save(path: Path, data: dict) -> None:
    save_json_atomic(path, data)


def _is_active(record: dict, now: datetime) -> bool:
    try:
        return isinstance(record, dict) and now < _parse_iso(record["expires_at"])
    except (KeyError, ValueError, TypeError):
        return False


def _operation_lock_path(path: Path, model: str) -> Path:
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    return path.with_suffix(path.suffix + f".operation-{digest}.lock")


def _try_operation_lock(path: Path, model: str) -> int | None:
    lock_path = _operation_lock_path(path, model)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
    try:
        _try_lock(fd)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _release_operation_lock(fd: int) -> None:
    try:
        _unlock(fd)
    finally:
        os.close(fd)


def _operation_is_live(path: Path, record: dict) -> bool:
    operation_id = record.get("operation_id") if isinstance(record, dict) else None
    if operation_id in _OPERATION_LOCKS:
        return True
    try:
        model = canonical_model(record.get("model"))
    except (AttributeError, ValueError):
        return False
    fd = _try_operation_lock(path, model)
    if fd is None:
        return True
    _release_operation_lock(fd)
    return False


def _prune_expired(data: dict, now: datetime, path: Path) -> None:
    data["claims"] = [r for r in data["claims"] if _is_active(r, now)]
    # An elapsed wall-clock lease does not evict a still-live operation owner.
    # The per-operation OS lock proves liveness across processes; an absent lock
    # means a crashed holder's bounded lease is safe to discard.
    data["operations"] = [
        r for r in data["operations"]
        if _is_active(r, now) or _operation_is_live(path, r)
    ]


def _canonical_record(record: dict) -> dict | None:
    if not isinstance(record, dict):
        return None
    if record.get("kind") == "reservation":
        return dict(record)
    try:
        model = canonical_model(record.get("model"))
    except ValueError:
        return None
    return {**record, "model": model}


def _active_model_claims(data: dict, model: str, now: datetime) -> list[dict]:
    result = []
    for record in data["claims"]:
        if not _is_active(record, now):
            continue
        canonical = _canonical_record(record)
        if canonical is not None and canonical.get("model") == model:
            result.append(canonical)
    return result


def _pending_for(data: dict, model: str, now: datetime) -> list[dict]:
    result = []
    for record in data["operations"]:
        try:
            same_model = canonical_model(record.get("model")) == model
        except (AttributeError, ValueError):
            same_model = False
        if same_model:
            result.append(dict(record))
    return result


def _pending_warm_for_scope(data: dict, scope: str) -> list[dict]:
    """Warm operations reserve admission capacity for their whole GPU scope."""
    return [dict(record) for record in data["operations"]
            if record.get("kind") == "warm" and _operation_scope(record) == scope]


def _has_pending_warm(data: dict) -> bool:
    return any(record.get("kind") == "warm" for record in data["operations"])


def claim(
    model: str, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Register a model claim unless an operation has already begun on it."""
    path = path or _DEFAULT_PATH
    model = canonical_model(model)
    owner = nonblank_text(owner, "owner")
    purpose = nonblank_text(purpose, "purpose")
    ttl_seconds = positive_ttl_seconds(ttl_seconds)
    now = now_fn()
    expires_at = _expires_at(now, ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "model", "model": model,
        "owner": owner, "purpose": purpose, "claimed_at": _iso(now),
        "renewed_at": _iso(now), "ttl_seconds": ttl_seconds,
        "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        if _pending_for(data, model, now):
            raise ValueError(f"operation pending for {model}")
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"],
            "model": model}


def reserve(
    gb: float, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    pid: Optional[int] = None, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Reserve finite positive VRAM capacity for a finite positive TTL."""
    path = path or _DEFAULT_PATH
    gb = positive_gb(gb)
    owner = nonblank_text(owner, "owner")
    purpose = nonblank_text(purpose, "purpose")
    ttl_seconds = positive_ttl_seconds(ttl_seconds)
    now = now_fn()
    expires_at = _expires_at(now, ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "reservation", "model": None,
        "gb": gb, "pid": pid, "owner": owner, "purpose": purpose,
        "claimed_at": _iso(now), "renewed_at": _iso(now),
        "ttl_seconds": ttl_seconds, "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        if _has_pending_warm(data):
            raise ValueError("warm admission pending; retry reservation after it completes")
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}


def renew(
    claim_id: str, ttl_seconds: Optional[int] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Extend a live claim; malformed legacy TTLs are rejected, never guessed."""
    path = path or _DEFAULT_PATH
    claim_id = nonblank_text(claim_id, "claim_id")
    requested_ttl = positive_ttl_seconds(ttl_seconds) if ttl_seconds is not None else None
    now = now_fn()
    with _locked(path):
        data = _load(path)
        for record in data["claims"]:
            if isinstance(record, dict) and record.get("claim_id") == claim_id:
                if not _is_active(record, now):
                    return {"ok": False}
                ttl = requested_ttl if requested_ttl is not None else positive_ttl_seconds(
                    record.get("ttl_seconds"))
                record["ttl_seconds"] = ttl
                record["renewed_at"] = _iso(now)
                record["expires_at"] = _iso(_expires_at(now, ttl))
                _prune_expired(data, now, path)
                _save(path, data)
                return {"ok": True, "expires_at": record["expires_at"]}
    return {"ok": False}


def release(
    claim_id: str, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Remove a named claim and prune expired records under the same lock."""
    path = path or _DEFAULT_PATH
    claim_id = nonblank_text(claim_id, "claim_id")
    now = now_fn()
    with _locked(path):
        data = _load(path)
        before = len(data["claims"])
        data["claims"] = [
            r for r in data["claims"]
            if not (isinstance(r, dict) and r.get("claim_id") == claim_id)
        ]
        found = len(data["claims"]) != before
        if found:
            _prune_expired(data, now, path)
            _save(path, data)
    return {"ok": found}


def list_claims(
    model: Optional[str] = None, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> list[dict]:
    """Return active claims, raising if an existing ledger cannot be trusted."""
    path = path or _DEFAULT_PATH
    target = canonical_model(model) if model is not None else None
    now = now_fn()
    data = _load(path)
    active = []
    for record in data["claims"]:
        if not _is_active(record, now):
            continue
        canonical = _canonical_record(record)
        if canonical is None:
            continue
        if target is None or canonical.get("model") == target:
            active.append(canonical)
    return active


def begin_operation(
    model: str, kind: str, force: bool = False, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now, scope: str = _DEFAULT_SCOPE,
) -> dict:
    """Atomically reserve a same-model operation lease before backend work.

    A 120-second lease is conservative but bounded: a crashed caller can delay
    a same-model request until expiry, rather than permanently block the ledger.
    """
    path = path or _DEFAULT_PATH
    model = canonical_model(model)
    kind = nonblank_text(kind, "kind")
    scope = nonblank_text(scope, "scope")
    now = now_fn()
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        pending = _pending_for(data, model, now)
        if pending:
            return {"ok": False, "outcome": "refused", "reason": "operation_pending",
                    "model": model, "operations": pending}
        if kind == "warm":
            capacity_pending = _pending_warm_for_scope(data, scope)
            if capacity_pending:
                return {"ok": False, "outcome": "refused", "reason": "capacity_pending",
                        "model": model, "scope": scope, "operations": capacity_pending}
        active_claims = _active_model_claims(data, model, now)
        if kind in _EVICTION_KINDS and not force and active_claims:
            return {"ok": False, "outcome": "refused", "reason": "model_claimed",
                    "model": model, "claims": active_claims}
        fd = _try_operation_lock(path, model)
        if fd is None:
            return {"ok": False, "outcome": "refused", "reason": "operation_pending",
                    "model": model, "operations": []}
        try:
            operation_id = uuid.uuid4().hex
            expires_at = _expires_at(now, _OPERATION_LEASE_SECONDS)
            record = {"operation_id": operation_id, "model": model, "kind": kind,
                      "scope": scope, "started_at": _iso(now), "expires_at": _iso(expires_at)}
            data["operations"].append(record)
            _save(path, data)
            _OPERATION_LOCKS[operation_id] = fd
        except Exception:
            _release_operation_lock(fd)
            raise
    return {"ok": True, "outcome": "begun", "operation_id": operation_id,
            "model": model, "kind": kind, "scope": scope, "expires_at": record["expires_at"]}


def finish_operation(
    operation_id: str, uncertain: bool = False, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Clear an operation lease, or retain it after unknown backend outcome.

    ``uncertain=True`` releases this process's live owner lock but preserves the
    bounded durable lease, so a timed-out request does not immediately permit a
    conflicting same-model mutation.
    """
    path = path or _DEFAULT_PATH
    operation_id = nonblank_text(operation_id, "operation_id")
    now = now_fn()
    expires_at = None
    try:
        with _locked(path):
            data = _load(path)
            _prune_expired(data, now, path)
            found = any(isinstance(r, dict) and r.get("operation_id") == operation_id
                        for r in data["operations"])
            if found and uncertain:
                expires_at = _iso(_expires_at(now, _OPERATION_LEASE_SECONDS))
                for record in data["operations"]:
                    if isinstance(record, dict) and record.get("operation_id") == operation_id:
                        record["expires_at"] = expires_at
                _save(path, data)
            elif found:
                data["operations"] = [
                    r for r in data["operations"]
                    if not (isinstance(r, dict) and r.get("operation_id") == operation_id)
                ]
                _save(path, data)
    finally:
        fd = _OPERATION_LOCKS.pop(operation_id, None)
        if fd is not None:
            _release_operation_lock(fd)
    result = {"ok": found, "retained": bool(found and uncertain)}
    if found and uncertain:
        result["expires_at"] = expires_at
    return result
