"""Map an Ollama model name to its ``llama-server`` runner PID.

Ollama spawns one ``llama-server`` OS subprocess per loaded model but exposes
no PID in its own API (``/api/ps`` has no ``pid`` field, confirmed against
Ollama 0.31.1). The only correlation path: read the runner's ``--model
<path>`` command-line argument (a blob file named ``sha256-<digest>``), then
search Ollama's on-disk manifests for the one whose model-layer digest
matches, which reveals the model's tag.

This is inherently undocumented and version-dependent. Every function
degrades to ``None`` on any parse/lookup failure rather than guessing, so a
future Ollama layout change only turns a model's ``busy`` signal into
"unknown," never breaks anything else.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

_MODEL_FLAG_RE = re.compile(r"--model\s+(\S+)")
_BLOB_DIGEST_RE = re.compile(r"sha256-([0-9a-f]{64})", re.IGNORECASE)
_MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _default_manifests_root() -> Path:
    models_dir = os.environ.get("OLLAMA_MODELS")
    if models_dir:
        return Path(models_dir) / "manifests"
    return Path.home() / ".ollama" / "models" / "manifests"


def _list_llama_server_processes_windows(timeout: int = 5) -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via one ``wmic`` call. ``[]`` on failure."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='llama-server.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, OSError):
        return []
    out = []
    lines = [ln.rstrip() for ln in result.stdout.splitlines() if ln.strip()]
    for line in lines[1:]:  # skip the header row
        parts = line.rsplit(None, 1)  # PID is the trailing whitespace-separated token
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        out.append({"pid": int(parts[1]), "cmdline": parts[0].strip()})
    return out


def _list_llama_server_processes_posix(timeout: int = 5) -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via ``ps``. ``[]`` on failure."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, OSError):
        return []
    out = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if "llama-server" not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        out.append({"pid": int(parts[0]), "cmdline": parts[1]})
    return out


def _list_llama_server_processes(timeout: int = 5) -> list[dict]:
    if sys.platform == "win32":
        return _list_llama_server_processes_windows(timeout)
    return _list_llama_server_processes_posix(timeout)


def _extract_model_digest(cmdline: str) -> Optional[str]:
    """Pull the sha256 hex digest out of a runner's ``--model <blob-path>`` arg.

    Requires exactly 64 hex characters (a real sha256 digest) — a shorter or
    malformed match is never accepted, so this never returns a partial guess.
    """
    m = _MODEL_FLAG_RE.search(cmdline)
    if not m:
        return None
    blob_match = _BLOB_DIGEST_RE.search(m.group(1))
    return blob_match.group(1).lower() if blob_match else None


def _resolve_tag_for_digest(digest: str, manifests_root: Path) -> Optional[str]:
    """Search every manifest file for one whose model layer matches ``digest``.

    A manifest's path relative to ``manifests_root`` is always
    ``<registry-host>/<namespace>/<name>/<tag>`` — Ollama's on-disk layout
    (e.g. ``registry.ollama.ai/library/qwen3/8b``, verified live). The tag
    reported is ``name:tag`` for the ``library`` namespace (matching
    Ollama's own ``/api/ps`` naming), or ``namespace/name:tag`` otherwise.
    A path that isn't exactly 4 levels deep is skipped, not guessed at.
    """
    if not manifests_root.is_dir():
        return None
    target = f"sha256:{digest}"
    for manifest_path in manifests_root.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = doc.get("layers", [])
        match = any(
            layer.get("mediaType") == _MODEL_LAYER_MEDIA_TYPE and layer.get("digest") == target
            for layer in layers
        )
        if not match:
            continue
        rel = manifest_path.relative_to(manifests_root).parts
        if len(rel) != 4:
            continue
        _registry_host, namespace, name, tag = rel
        return f"{name}:{tag}" if namespace == "library" else f"{namespace}/{name}:{tag}"
    return None


def find_pid_for_model(
    model_name: str, *,
    list_processes=None,
    manifests_root: Optional[Path] = None,
) -> Optional[int]:
    """The OS PID of the ``llama-server`` runner currently serving ``model_name``.

    ``None`` if Ollama isn't running that model, or if correlation fails for
    any reason (unexpected command-line shape, manifest missing/unparsable).
    """
    list_processes = list_processes or _list_llama_server_processes
    manifests_root = manifests_root or _default_manifests_root()
    for proc in list_processes():
        digest = _extract_model_digest(proc["cmdline"])
        if not digest:
            continue
        if _resolve_tag_for_digest(digest, manifests_root) == model_name:
            return proc["pid"]
    return None
