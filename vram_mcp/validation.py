"""Validation for values that become durable coordination records."""

from __future__ import annotations

import math
import sys


def nonblank_text(value, name: str) -> str:
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError(f"{name} must be a non-blank string")
    return text


def positive_gb(value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"gb must be a positive finite number, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"gb must be a positive finite number, got {value!r}")
    if (not math.isfinite(value) or value <= 0
            or value > sys.maxsize / 1024):
        raise ValueError(f"gb must be a positive finite number, got {value!r}")
    return value


def positive_ttl_seconds(value) -> int:
    """Validate a TTL that survives the ledger's whole-second ISO format."""
    if isinstance(value, bool):
        raise ValueError("ttl_seconds must be a positive whole number of seconds")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("ttl_seconds must be a positive whole number of seconds")
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError("ttl_seconds must be a positive whole number of seconds")
    return int(number)
