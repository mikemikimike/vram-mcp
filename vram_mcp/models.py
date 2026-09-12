"""Canonical names for Ollama models used as coordination keys."""

from __future__ import annotations


def canonical_model(model: str) -> str:
    """Return a case-insensitive Ollama coordination key.

    This follows Ollama's name defaults and case-insensitive comparison:
    https://github.com/ollama/ollama/blob/main/types/model/name.go
    Untagged names gain ``:latest``; the default registry and ``library``
    namespace are removed. A registry port is only meaningful before the final
    path component, so it never creates a model tag.
    """
    if not isinstance(model, str):
        raise ValueError("model must be a non-blank string")
    model = model.strip()
    if not model:
        raise ValueError("model must be a non-blank string")
    lowered = model.lower()
    for scheme in ("https://", "http://"):
        if lowered.startswith(scheme):
            model = model[len(scheme):]
            break
    parts = model.split("/")
    if any(not part.strip() or part != part.strip() for part in parts):
        raise ValueError("model must not contain blank path segments")
    parts = [part.lower() for part in parts]
    if len(parts) == 3 and parts[0] == "registry.ollama.ai":
        parts.pop(0)
    if len(parts) == 2 and parts[0] == "library":
        parts.pop(0)
    if not parts:
        raise ValueError("model must name a repository")
    final = parts[-1]
    if ":" in final:
        name, tag = final.rsplit(":", 1)
        if not name or not tag or ":" in name:
            raise ValueError("model tag must be non-blank")
    else:
        parts[-1] = f"{final}:latest"
    return "/".join(parts)
