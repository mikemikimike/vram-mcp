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

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ._util import (
    locked as _locked, iso as _iso, parse_iso as _parse_iso,
    save_json_atomic, load_json,
)

_DEFAULT_PATH = Path.home() / ".cache" / "vram-mcp" / "claims.json"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _load(path: Path) -> dict:
    return load_json(
        path, lambda: {"claims": []},
        lambda d: isinstance(d, dict) and isinstance(d.get("claims"), list),
    )


def _save(path: Path, data: dict) -> None:
    save_json_atomic(path, data)


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
        "kind": "model",
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


def reserve(
    gb: float, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    pid: Optional[int] = None, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Reserve ``gb`` GB of VRAM for ``owner`` — a claim on capacity rather
    than on a named model.

    The GPU's biggest consumer is often not an Ollama model (a training run, a
    diffusion job), and such a process has no other way to tell other sessions
    its VRAM is spoken for. Reservations share the ledger with model claims, so
    they inherit the same TTL and crash-safety semantics, tagged
    ``kind="reservation"`` and carrying ``model=None`` so they can never shadow
    or protect a model. ``pid`` is advisory: it records which process the
    reservation is for.

    Returns ``{"claim_id", "expires_at"}``.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    expires_at = now + timedelta(seconds=ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "reservation",
        "model": None, "gb": float(gb), "pid": pid,
        "owner": owner, "purpose": purpose,
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


def release(
    claim_id: str, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Remove a claim immediately (before its TTL expires).

    ``now_fn`` matches the other write paths (clock injection for tests) and
    drives the same expired-record pruning every locked write performs.
    """
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
            _prune_expired(data, now_fn())
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

    The ledger holds both kinds of record. Reservations (``kind="reservation"``,
    ``model=None``) are returned by an unfiltered call — that's how a caller
    totals the VRAM other sessions have spoken for — and are always excluded by
    a ``model=`` filter, since a reservation names no model. Records written by
    older versions carry no ``kind`` at all and are treated as model claims,
    exactly as before; the filter uses ``.get("model")`` so a record missing the
    key entirely is skipped rather than raising.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    active = [r for r in _load(path)["claims"] if _is_active(r, now)]
    if model is not None:
        active = [r for r in active if r.get("model") == model]
    return active
