"""FastMCP server exposing VRAM-management tools.

This is the only module that imports ``mcp``. It wires the pure logic in
:mod:`vram_mcp.core` / :mod:`vram_mcp.gpu` / :mod:`vram_mcp.ollama` /
:mod:`vram_mcp.nvml` / :mod:`vram_mcp.ollama_correlate` /
:mod:`vram_mcp.claims` to MCP tools.

Every tool is ``async def`` and runs its blocking body (subprocess spawns,
NVML sessions, HTTP calls, file locks) in a worker thread via
``anyio.to_thread.run_sync`` — the installed FastMCP invokes sync tools
directly on the asyncio event loop, so a plain ``def`` tool would block the
whole server (pings included) for the duration of every nvidia-smi/wmic call.
"""

from __future__ import annotations

import functools
import os

from typing import Optional

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from . import claims as _claims
from . import core
from . import nvml as _nvml
from .gpu import gpu_status
from .ollama import OllamaClient
from .ollama_correlate import runner_pid_map

mcp = FastMCP("vram-mcp")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)


def _fmt_free(free_mb) -> str:
    return "unknown (nvidia-smi unavailable)" if free_mb is None else f"{free_mb} MB"


def _snapshot() -> core.Snapshot:
    """One capture of the coordination signals (claims + pid map + busy map),
    shared across every model an operation touches."""
    return core.Snapshot.capture(
        _claims.list_claims, runner_pid_map, _nvml.nvml_busy_map,
    )


def _full_status() -> dict:
    """The enriched combined status both status tools share."""
    return core.combined_status(
        gpu_status, _ollama,
        snapshot_fn=_snapshot,
        nvml_processes_fn=_nvml.nvml_processes,
    )


async def _in_thread(fn, *args, **kwargs):
    """Run a blocking tool body off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


# ── status tools ─────────────────────────────────────────────────────────────

def _vram_status_impl() -> dict:
    status = _full_status()
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"])
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}."
    )
    return status


@mcp.tool()
async def vram_status() -> dict:
    """Report GPU VRAM and currently loaded Ollama models.

    Returns per-GPU totals, the list of resident models (each with claim
    attribution, a best-effort busy signal, and CPU-offload detection), every
    other VRAM-holding process on the GPU, the best free VRAM, and a
    human-readable ``summary``.
    """
    return await _in_thread(_vram_status_impl)


def _list_loaded_impl() -> dict:
    loaded = _full_status()["loaded"]
    return {
        "loaded": loaded,
        "summary": f"{len(loaded)} model(s) loaded.",
    }


@mcp.tool()
async def list_loaded() -> dict:
    """List the models currently resident in VRAM, with claim/busy detail."""
    return await _in_thread(_list_loaded_impl)


# ── eviction tools ───────────────────────────────────────────────────────────

def _unload_impl(model: str, force: bool) -> dict:
    if not force:
        protected, detail = core.is_protected(model, _snapshot())
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
async def unload(model: str, force: bool = False) -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort
    ``busy`` signal — pass ``force=True`` to override.
    """
    return await _in_thread(_unload_impl, model, force)


def _ensure_free_impl(gb: float, force: bool) -> dict:
    result = core.ensure_free(
        gb, gpu_status, _ollama, settle=0.5, force=force,
        snapshot_fn=_snapshot,
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
async def ensure_free(gb: float, force: bool = False) -> dict:
    """Free VRAM until at least ``gb`` gigabytes are available.

    Unloads resident models largest-first until the target is met, skipping
    claimed/busy models by default (``force=True`` to override). Returns
    which models were unloaded, which were declined (protected), and whether
    the target was reached.
    """
    return await _in_thread(_ensure_free_impl, gb, force)


def _warm_impl(model: str, keep_alive: str) -> dict:
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
async def warm(model: str, keep_alive: str = "5m") -> dict:
    """Load/pin a model into VRAM for ``keep_alive`` (e.g. ``"5m"``, ``"1h"``)."""
    return await _in_thread(_warm_impl, model, keep_alive)


# ── claim tools ──────────────────────────────────────────────────────────────
# The ledger can raise on genuinely-contended/odd filesystem states
# (TimeoutError from a live lock held past the wait window; OSError if the
# Windows sharing-violation retry in claims._save is exhausted). Those are
# operational outcomes, not bugs — surface them as structured {ok: false}
# responses instead of raw tracebacks.

def _claim_impl(model: str, owner: str, purpose: str, ttl_seconds: int) -> dict:
    try:
        result = _claims.claim(model, owner, purpose, ttl_seconds)
    except (TimeoutError, OSError) as e:
        return {"ok": False, "summary": f"Claim failed: {e}"}
    result["ok"] = True
    result["summary"] = (
        f"Claimed '{model}' for {owner} ({purpose}), expires {result['expires_at']}."
    )
    return result


@mcp.tool()
async def claim(model: str, owner: str, purpose: str, ttl_seconds: int = 3600) -> dict:
    """Declare that you're using ``model`` for ``purpose``.

    Lets other sessions see who's using a model and why before deciding to
    evict it. Renew before ``ttl_seconds`` elapses if still in use — an
    un-renewed claim simply expires, so a crashed session never leaves a
    permanently-stuck claim.
    """
    return await _in_thread(_claim_impl, model, owner, purpose, ttl_seconds)


def _renew_impl(claim_id: str, ttl_seconds: Optional[int]) -> dict:
    try:
        result = _claims.renew(claim_id, ttl_seconds)
    except (TimeoutError, OSError) as e:
        return {"ok": False, "summary": f"Renew failed: {e}"}
    result["summary"] = (
        f"Renewed, expires {result['expires_at']}." if result["ok"]
        else "No such claim (already expired or released?)."
    )
    return result


@mcp.tool()
async def renew(claim_id: str, ttl_seconds: Optional[int] = None) -> dict:
    """Extend an existing claim's expiry before it lapses."""
    return await _in_thread(_renew_impl, claim_id, ttl_seconds)


def _release_impl(claim_id: str) -> dict:
    try:
        result = _claims.release(claim_id)
    except (TimeoutError, OSError) as e:
        return {"ok": False, "summary": f"Release failed: {e}"}
    result["summary"] = "Released." if result["ok"] else "No such claim."
    return result


@mcp.tool()
async def release(claim_id: str) -> dict:
    """Release a claim early, before its TTL would expire."""
    return await _in_thread(_release_impl, claim_id)


def _list_claims_impl(model: Optional[str]) -> dict:
    active = _claims.list_claims(model)
    return {"claims": active, "summary": f"{len(active)} active claim(s)."}


@mcp.tool()
async def list_claims(model: Optional[str] = None) -> dict:
    """See who's claiming what right now (all models, or one)."""
    return await _in_thread(_list_claims_impl, model)


# ── advice ───────────────────────────────────────────────────────────────────

def _advise_impl() -> dict:
    result = core.advise(gpu_status, _ollama)
    n = len(result["suggestions"])
    result["summary"] = (
        "No VRAM issues detected." if n == 0 else f"{n} suggestion(s)."
    )
    return result


@mcp.tool()
async def advise() -> dict:
    """Suggest env/config changes to keep VRAM healthy (heuristics)."""
    return await _in_thread(_advise_impl)


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
