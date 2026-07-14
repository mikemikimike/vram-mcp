"""FastMCP server exposing VRAM-management tools.

This is the only module that imports ``mcp``. It wires the pure logic in
:mod:`vram_mcp.core` / :mod:`vram_mcp.gpu` / :mod:`vram_mcp.ollama` to MCP tools.
"""

from __future__ import annotations

import os

from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import claims as _claims
from . import core
from . import nvml as _nvml
from .gpu import gpu_status
from .ollama import OllamaClient
from .ollama_correlate import find_pid_for_model as _find_pid_for_model

mcp = FastMCP("vram-mcp")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)


def _fmt_free(free_mb) -> str:
    return "unknown (nvidia-smi unavailable)" if free_mb is None else f"{free_mb} MB"


def _list_claims_fn(model: str) -> list[dict]:
    return _claims.list_claims(model)


def _busy_fn(pid) -> Optional[bool]:
    return _nvml.nvml_busy(pid)


def _nvml_processes_fn() -> list[dict]:
    return _nvml.nvml_processes()


@mcp.tool()
def vram_status() -> dict:
    """Report GPU VRAM and currently loaded Ollama models.

    Returns per-GPU totals, the list of resident models (each with claim
    attribution, a best-effort busy signal, and CPU-offload detection), every
    other VRAM-holding process on the GPU, the best free VRAM, and a
    human-readable ``summary``.
    """
    status = core.combined_status(
        gpu_status, _ollama,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
        nvml_processes_fn=_nvml_processes_fn,
    )
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"])
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}."
    )
    return status


@mcp.tool()
def list_loaded() -> dict:
    """List the models currently resident in VRAM, with claim/busy detail."""
    status = core.combined_status(
        gpu_status, _ollama,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
        nvml_processes_fn=_nvml_processes_fn,
    )
    loaded = status["loaded"]
    return {
        "loaded": loaded,
        "summary": f"{len(loaded)} model(s) loaded.",
    }


@mcp.tool()
def unload(model: str, force: bool = False) -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort
    ``busy`` signal — pass ``force=True`` to override.
    """
    if not force:
        protected, detail = core.is_protected(
            model, _list_claims_fn, _find_pid_for_model, _busy_fn,
        )
        if protected:
            return {
                "ok": False,
                "model": model,
                "protected": True,
                **detail,
                "summary": (
                    f"'{model}' is protected (claimed or busy); "
                    "pass force=True to override."
                ),
            }
    ok = _ollama.unload(model)
    return {
        "ok": ok,
        "model": model,
        "summary": (
            f"Unloaded '{model}'." if ok else f"Failed to unload '{model}'."
        ),
    }


@mcp.tool()
def ensure_free(gb: float, force: bool = False) -> dict:
    """Free VRAM until at least ``gb`` gigabytes are available.

    Unloads resident models largest-first until the target is met, skipping
    claimed/busy models by default (``force=True`` to override). Returns
    which models were unloaded, which were declined (protected), and whether
    the target was reached.
    """
    result = core.ensure_free(
        gb, gpu_status, _ollama, settle=0.5, force=force,
        list_claims_fn=_list_claims_fn,
        find_pid_fn=_find_pid_for_model,
        busy_fn=_busy_fn,
    )
    declined_note = ""
    if result["declined"]:
        names = ", ".join(d["name"] for d in result["declined"])
        declined_note = f" Protected (force=True to override): {names}."
    if result["already_free"]:
        result["summary"] = (
            f"Already {_fmt_free(result['free_mb'])} free (target {gb} GB)."
        )
    elif result["ok"]:
        result["summary"] = (
            f"Freed VRAM to {_fmt_free(result['free_mb'])} "
            f"(target {gb} GB) by unloading: "
            f"{', '.join(result['unloaded']) or 'none'}."
        ) + declined_note
    else:
        result["summary"] = (
            f"Could not reach {gb} GB free "
            f"(now {_fmt_free(result['free_mb'])}); "
            f"unloaded: {', '.join(result['unloaded']) or 'none'}."
        ) + declined_note
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
def claim(model: str, owner: str, purpose: str, ttl_seconds: int = 3600) -> dict:
    """Declare that you're using ``model`` for ``purpose``.

    Lets other sessions see who's using a model and why before deciding to
    evict it. Renew before ``ttl_seconds`` elapses if still in use — an
    un-renewed claim simply expires, so a crashed session never leaves a
    permanently-stuck claim.
    """
    result = _claims.claim(model, owner, purpose, ttl_seconds)
    result["summary"] = (
        f"Claimed '{model}' for {owner} ({purpose}), expires {result['expires_at']}."
    )
    return result


@mcp.tool()
def renew(claim_id: str, ttl_seconds: Optional[int] = None) -> dict:
    """Extend an existing claim's expiry before it lapses."""
    result = _claims.renew(claim_id, ttl_seconds)
    result["summary"] = (
        f"Renewed, expires {result['expires_at']}." if result["ok"]
        else "No such claim (already expired or released?)."
    )
    return result


@mcp.tool()
def release(claim_id: str) -> dict:
    """Release a claim early, before its TTL would expire."""
    result = _claims.release(claim_id)
    result["summary"] = "Released." if result["ok"] else "No such claim."
    return result


@mcp.tool()
def list_claims(model: Optional[str] = None) -> dict:
    """See who's claiming what right now (all models, or one)."""
    active = _claims.list_claims(model)
    return {"claims": active, "summary": f"{len(active)} active claim(s)."}


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
