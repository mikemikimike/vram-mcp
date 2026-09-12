"""Small shared helpers with no dependencies on any sibling module."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BYTES_PER_MB = 1024 * 1024


def bytes_to_mb(value, default=None) -> Optional[int]:
    """Best-effort bytes → whole MB; ``default`` on missing/garbage input.

    Callers pick the fallback that matches their contract: ``0`` where a
    number is always expected (Ollama sizes), ``None`` where "unreported"
    is meaningful (NVML per-process sizes on Windows/WDDM).
    """
    try:
        return int(value) // BYTES_PER_MB
    except (TypeError, ValueError, OverflowError):
        return default


def run_capture(cmd: list[str], timeout: int) -> Optional[str]:
    """Run ``cmd`` and return stdout, or ``None`` on ANY failure (missing
    binary, timeout, non-zero exit). For callers whose contract is
    best-effort telemetry — never raises."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        # FileNotFoundError is an OSError subclass — one tuple covers both.
        return None
    return result.stdout


_REPLACE_ATTEMPTS = 10
_REPLACE_RETRY_SLEEP = 0.02


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def locked(path: Path, timeout: float = 5.0, poll: float = 0.05):
    """Serialize access through an OS-backed sibling lock file.

    The file is deliberately persistent.  Its *advisory lock*, rather than
    its age or existence, represents ownership; the OS releases it when a
    crashed holder exits.  This avoids both stale-file split brain on POSIX and
    sharing violations from deleting an open lock on Windows.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        # Windows locks byte ranges and cannot lock an empty file.
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        while True:
            try:
                _try_lock(fd)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not acquire lock {lock_path}")
                time.sleep(poll)
        yield
    finally:
        if acquired:
            try:
                _unlock(fd)
            except OSError:
                pass
        os.close(fd)


def _try_lock(fd: int) -> None:
    """Take a non-blocking exclusive advisory lock, or raise BlockingIOError."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _quarantine(path: Path) -> None:
    try:
        os.replace(path, path.with_suffix(path.suffix + ".corrupt"))
    except OSError:
        pass


def load_json(path: Path, default_factory, is_valid):
    """Load JSON; on missing/unparsable/invalid-shape return ``default_factory()``.
    A file that EXISTS but is corrupt/wrong-shape is quarantined to ``.corrupt``
    first, so the next write can't silently destroy it."""
    if not path.exists():
        return default_factory()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
        _quarantine(path)
        return default_factory()
    if not is_valid(doc):
        _quarantine(path)
        return default_factory()
    return doc


def save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(_REPLACE_RETRY_SLEEP)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_jsonl_capped(path: Path, record: dict, cap: int) -> None:
    """Append one JSON line; if the file then exceeds ``cap`` lines, rewrite it
    to the last ``cap``. Bounded growth, no daemon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    rows = read_jsonl(path)
    if len(rows) > cap:
        kept = rows[-cap:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return  # best-effort prune; a failed prune never breaks logging
                time.sleep(_REPLACE_RETRY_SLEEP)
