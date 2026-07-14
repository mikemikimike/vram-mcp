"""FastMCP server exposing VRAM-management tools.

This is the only module that imports ``mcp``. It wires the pure logic in
:mod:`vram_mcp.core` / :mod:`vram_mcp.gpu` / :mod:`vram_mcp.ollama` to MCP tools.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from . import core
from .gpu import gpu_status
from .ollama import OllamaClient

mcp = FastMCP("vram-mcp")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)


def _fmt_free(free_mb) -> str:
    return "unknown (nvidia-smi unavailable)" if free_mb is None else f"{free_mb} MB"


@mcp.tool()
def vram_status() -> dict:
    """Report GPU VRAM and currently loaded Ollama models.

    Returns per-GPU totals, the list of resident models, the best free VRAM,
    and a human-readable ``summary``.
    """
    status = core.combined_status(gpu_status, _ollama)
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"])
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}."
    )
    return status


@mcp.tool()
def list_loaded() -> dict:
    """List the models currently resident in VRAM (name, VRAM MB, expiry)."""
    loaded = core.combined_status(gpu_status, _ollama)["loaded"]
    return {
        "loaded": loaded,
        "summary": f"{len(loaded)} model(s) loaded.",
    }


@mcp.tool()
def unload(model: str) -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``)."""
    ok = _ollama.unload(model)
    return {
        "ok": ok,
        "model": model,
        "summary": (
            f"Unloaded '{model}'." if ok else f"Failed to unload '{model}'."
        ),
    }


@mcp.tool()
def ensure_free(gb: float) -> dict:
    """Free VRAM until at least ``gb`` gigabytes are available.

    Unloads resident models largest-first until the target is met. Returns
    which models were unloaded and whether the target was reached.
    """
    result = core.ensure_free(gb, gpu_status, _ollama, settle=0.5)
    if result["already_free"]:
        result["summary"] = (
            f"Already {_fmt_free(result['free_mb'])} free (target {gb} GB)."
        )
    elif result["ok"]:
        result["summary"] = (
            f"Freed VRAM to {_fmt_free(result['free_mb'])} "
            f"(target {gb} GB) by unloading: "
            f"{', '.join(result['unloaded']) or 'none'}."
        )
    else:
        result["summary"] = (
            f"Could not reach {gb} GB free "
            f"(now {_fmt_free(result['free_mb'])}); "
            f"unloaded: {', '.join(result['unloaded']) or 'none'}."
        )
    return result


@mcp.tool()
def warm(model: str, keep_alive: str = "5m") -> dict:
    """Load/pin a model into VRAM for ``keep_alive`` (e.g. ``"5m"``, ``"1h"``)."""
    ok = _ollama.warm(model, keep_alive)
    return {
        "ok": ok,
        "model": model,
        "keep_alive": keep_alive,
        "summary": (
            f"Warmed '{model}' (keep_alive={keep_alive})."
            if ok
            else f"Failed to warm '{model}'."
        ),
    }


@mcp.tool()
def advise() -> dict:
    """Suggest env/config changes to keep VRAM healthy (heuristics)."""
    result = core.advise(gpu_status, _ollama)
    n = len(result["suggestions"])
    result["summary"] = (
        "No VRAM issues detected." if n == 0 else f"{n} suggestion(s)."
    )
    return result


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
