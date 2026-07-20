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

from datetime import datetime, timedelta, timezone
from typing import Optional

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from . import claims as _claims
from . import core
from . import nvml as _nvml
from ._util import iso
from .gpu import gpu_status
from .ollama import OllamaClient
from .ollama_correlate import resolve_tag, runner_pid_map

mcp = FastMCP("vram-mcp")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)

from . import audit as _audit
from . import procinfo as _procinfo

_AUDIT_ON = os.environ.get("VRAM_MCP_AUDIT", "1") != "0"
_MEANINGFUL_MB = int(os.environ.get("VRAM_MCP_MEANINGFUL_MB", "512"))
_EVENT_CAP = int(os.environ.get("VRAM_MCP_EVENT_CAP", "5000"))
_SAMPLE_SECONDS = float(os.environ.get("VRAM_MCP_SAMPLE_SECONDS", "60"))
_SPILL_MB = int(os.environ.get("VRAM_MCP_SPILL_MB", str(core.SPILL_THRESHOLD_MB)))
# trend() output lands in an agent's context window, so the raw rows are capped
# even though the SUMMARY is always computed over every row in the window.
_TREND_SAMPLE_CAP = 200


def _procinfo_table() -> list:
    """Sized+named process table; the Windows perf-counter fallback only fires
    when NVML sizes are null (inside win_gpu_procs, which no-ops off-Windows)."""
    return _procinfo.process_table(
        nvml_processes=_nvml.nvml_processes,
        win_gpu_reader=_procinfo.win_gpu_procs,
        posix_name_reader=_procinfo.posix_name_reader,
    )


def _run_detection(status: dict) -> None:
    """Diff the meaningful-holder set from a status snapshot and log changes,
    then record a throttled free-VRAM sample for ``trend``.

    Best-effort; disabled by VRAM_MCP_AUDIT=0 — never raises (the audit may
    never break a tool call). The two halves are guarded independently so a
    sampling failure can't suppress detection, or vice versa.

    A status WITHOUT GPU rows (``list_loaded`` skips the nvidia-smi spawn on
    purpose) is never sampled: ``used_mb: 0``/``total_mb: 0`` would record a
    fiction as fact, and — worse — burn the shared once-per-interval throttle
    slot, so the next REAL sample from ``vram_status`` would be dropped.
    Detection still runs; only the sample is skipped."""
    if not _AUDIT_ON:
        return
    try:
        holders = _audit.meaningful_holders(
            status.get("loaded", []), status.get("other_processes", []), _MEANINGFUL_MB)
        _audit.detect_and_log(holders, cap=_EVENT_CAP)
    except Exception:
        pass
    gpus = status.get("gpus") or []
    if not gpus:
        return
    try:
        p = status.get("pressure") or {}
        _audit.maybe_log_sample(
            {
                "free_mb": p.get("free_mb"),
                "used_mb": sum(g.get("used_mb") or 0 for g in gpus),
                "total_mb": sum(g.get("total_mb") or 0 for g in gpus),
                "non_local_mb": p.get("non_local_mb"),
                "state": p.get("state"),
                "loaded_count": len(status.get("loaded", [])),
            },
            interval_seconds=_SAMPLE_SECONDS, cap=_EVENT_CAP,
        )
    except Exception:
        pass


def _fmt_free(free_mb) -> str:
    return "unknown (nvidia-smi unavailable)" if free_mb is None else f"{free_mb} MB"


def _snapshot() -> core.Snapshot:
    """One capture of the coordination signals (claims + pid map + busy map),
    shared across every model an operation touches."""
    return core.Snapshot.capture(
        _claims.list_claims, runner_pid_map, _nvml.nvml_busy_map,
    )


def _active_claims() -> tuple[list, bool]:
    """The ledger's active records for a read-only admission decision, plus
    whether the read actually succeeded.

    FAILS OPEN by design. ``list_claims`` is lock-free and swallows most IO
    errors, but a pathological filesystem state can still surface a
    TimeoutError/OSError, and the alternative — refusing to warm because a
    small JSON file was momentarily unreadable — is the worse failure: the
    user's GPU is idle, the refusal is inexplicable to them, and reservations
    were only ever cooperative advice in the first place. Warming past a
    reservation we couldn't read is recoverable; a GPU nobody can use is not.
    Callers get ``ok=False`` so they can say "unknown" rather than "zero"."""
    try:
        return _claims.list_claims(), True
    except (TimeoutError, OSError):
        return [], False


def _full_status() -> dict:
    """The enriched combined status both status tools share."""
    return core.combined_status(
        gpu_status, _ollama,
        snapshot_fn=_snapshot,
        procinfo_fn=(_procinfo_table if _AUDIT_ON else _nvml.nvml_processes),
        spill_threshold_mb=_SPILL_MB,
    )


async def _in_thread(fn, *args, **kwargs):
    """Run a blocking tool body off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


# ── status tools ─────────────────────────────────────────────────────────────

def _vram_status_impl() -> dict:
    status = _full_status()
    _run_detection(status)
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"])
    p = status.get("pressure") or {}
    state = p.get("state", "unknown")
    # The pressure detail is the actionable half ("X MB has spilled to system
    # RAM"), so surface it inline rather than making the caller dig into the
    # payload — but only when there's something to say, and only via .get() so
    # a partial pressure dict can never turn a status call into an exception.
    detail = p.get("detail")
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}, pressure: {state}."
        + (f" {detail}" if detail and state not in ("ok", "unknown") else "")
    )
    return status


@mcp.tool()
async def vram_status() -> dict:
    """Report GPU VRAM and currently loaded Ollama models.

    Returns per-GPU totals, the list of resident models (each with claim
    attribution, a best-effort busy signal, and CPU-offload detection), every
    other VRAM-holding process on the GPU, the best free VRAM, a ``pressure``
    dict (state ok|tight|degraded|thrashing, separating driver-forced spill to
    system RAM from Ollama's deliberate CPU offload), and a human-readable
    ``summary``.
    """
    return await _in_thread(_vram_status_impl)


def _list_loaded_impl() -> dict:
    # Slimmer than _full_status: this tool returns only the model list, so
    # skip the nvidia-smi spawn — but still gather procinfo (when audit is on)
    # so detection has a meaningful-holder set to diff against.
    status = core.combined_status(
        lambda: [], _ollama, snapshot_fn=_snapshot,
        procinfo_fn=(_procinfo_table if _AUDIT_ON else None),
        spill_threshold_mb=_SPILL_MB,
    )
    _run_detection(status)
    loaded = status["loaded"]
    return {
        "loaded": loaded,
        "summary": f"{len(loaded)} model(s) loaded.",
    }


@mcp.tool()
async def list_loaded() -> dict:
    """List the models currently resident in VRAM, with claim/busy detail."""
    return await _in_thread(_list_loaded_impl)


# ── eviction tools ───────────────────────────────────────────────────────────

def _unload_impl(model: str, force: bool, by: str) -> dict:
    if not force:
        protected, detail = core.is_protected(model, _snapshot())
        if protected:
            _audit.log_action(action="unload", target=model, kind="ollama",
                              actor=by, force=False, outcome="refused",
                              detail="protected (claimed or busy)", cap=_EVENT_CAP)
            summary = (
                f"'{model}' is protected (claimed or busy); "
                "pass force=True to override."
            )
            if detail["busy"] is True and not detail["claims"]:
                summary += (
                    " Note: busy reflects recent GPU activity and can lag a"
                    " few seconds after a generation ends — if the work you"
                    " know about has finished, force=True is safe."
                )
            return {
                "ok": False,
                "model": model,
                "protected": True,
                **detail,
                "summary": summary,
            }
    ok = _ollama.unload(model)
    _audit.log_action(action="unload", target=model, kind="ollama", actor=by,
                      force=force, outcome=("ok" if ok else "failed"),
                      detail=("unloaded" if ok else "ollama unload failed"),
                      cap=_EVENT_CAP)
    return {
        "ok": ok,
        "model": model,
        "summary": (
            f"Unloaded '{model}'." if ok else f"Failed to unload '{model}'."
        ),
    }


@mcp.tool()
async def unload(model: str, force: bool = False, by: str = "unknown") -> dict:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort busy
    signal — pass ``force=True`` to override (busy is windowed and can lag a few
    seconds past a generation). ``by`` records who requested the eviction in the
    audit log (see ``history``)."""
    return await _in_thread(_unload_impl, model, force, by)


def _ensure_free_impl(gb: float, force: bool, by: str) -> dict:
    result = core.ensure_free(
        gb, gpu_status, _ollama, settle=0.5, force=force,
        snapshot_fn=_snapshot,
    )
    active, ledger_ok = _active_claims()
    reserved = core.reserved_mb(active)
    # None, not 0, when the ledger couldn't be read — "we don't know" and
    # "nothing is reserved" lead to opposite decisions.
    result["reserved_mb"] = reserved if ledger_ok else None
    for name in result["unloaded"]:
        _audit.log_action(action="ensure_free", target=name, kind="ollama",
                          actor=by, force=force, outcome="ok",
                          detail=f"unloaded to free {gb} GB", cap=_EVENT_CAP)
    for d in result["declined"]:
        _audit.log_action(action="ensure_free", target=d["name"], kind="ollama",
                          actor=by, force=force, outcome="refused",
                          detail="protected (claimed or busy)", cap=_EVENT_CAP)
    if result["already_free"]:
        base = f"Already {_fmt_free(result['free_mb'])} free (target {gb} GB)."
    elif result["ok"]:
        base = (
            f"Freed VRAM to {_fmt_free(result['free_mb'])} "
            f"(target {gb} GB) by unloading: "
            f"{', '.join(result['unloaded']) or 'none'}."
        )
    else:
        base = (
            f"Could not reach {gb} GB free "
            f"(now {_fmt_free(result['free_mb'])}); "
            f"unloaded: {', '.join(result['unloaded']) or 'none'}."
        )
    declined_note = ""
    if result["declined"]:
        names = ", ".join(d["name"] for d in result["declined"])
        declined_note = f" Protected (force=True to override): {names}."
    # Freed VRAM isn't necessarily *yours*: another session may have reserved
    # part of it for non-Ollama work, and it won't show up as a loaded model.
    if not ledger_ok:
        reserved_note = (
            " Note: the claim ledger could not be read, so how much of the "
            "free VRAM is reserved by other sessions is unknown."
        )
    elif reserved:
        reserved_note = (
            f" Note: {reserved} MB of the free VRAM is reserved by other sessions."
        )
    else:
        reserved_note = ""
    result["summary"] = base + declined_note + reserved_note
    return result


@mcp.tool()
async def ensure_free(gb: float, force: bool = False, by: str = "unknown") -> dict:
    """Free VRAM until at least ``gb`` GB is available. Skips claimed/busy models
    unless ``force=True``. ``by`` records the requester in the audit log.

    Also reports ``reserved_mb`` — how much of the resulting free VRAM other
    sessions have reserved for non-Ollama work (``None`` if unreadable)."""
    return await _in_thread(_ensure_free_impl, gb, force, by)


def _warm_impl(model: str, keep_alive: str, by: str, force: bool) -> dict:
    if not force:
        status = core.combined_status(gpu_status, _ollama)
        # Fails open (see _active_claims): an unreadable ledger reads as "no
        # reservations", so the warm proceeds rather than being refused for a
        # reason the user can neither see nor fix.
        active, _ = _active_claims()
        reserved = core.reserved_mb(active)
        # Ollama resolves a bare name to its ":latest" tag, so a raw
        # tags().get(model) misses on exactly the names users type — and a miss
        # reads as "size unknown", which walks straight past the
        # insufficient_headroom refusal this whole feature exists for.
        sizes = _ollama.tags()
        tag = resolve_tag(model, sizes)
        allowed, detail = core.can_warm(
            model, free_mb=status["free_mb"], reserved_mb=reserved,
            model_size_mb=(sizes.get(tag) if tag is not None else None),
        )
        if not allowed:
            _audit.log_action(action="warm", target=model, kind="ollama",
                              actor=by, force=False, outcome="refused",
                              detail=f"reserved {reserved} MB ({detail['reason']})",
                              cap=_EVENT_CAP)
            owners = ", ".join(
                f"{r.get('owner')} ({r.get('gb')} GB, {r.get('purpose')})"
                for r in active if r.get("kind") == "reservation"
            )
            return {
                "ok": False, "model": model, "refused": True, **detail,
                "reservations": owners,
                "summary": (
                    f"Refused to warm '{model}': {reserved} MB of VRAM is "
                    f"reserved [{owners}] leaving {detail['headroom_mb']} MB "
                    "headroom. Pass force=True to override."
                ),
            }
    ok = _ollama.warm(model, keep_alive)
    _audit.log_action(action="warm", target=model, kind="ollama", actor=by,
                      force=force, outcome=("ok" if ok else "failed"),
                      detail=f"keep_alive={keep_alive}", cap=_EVENT_CAP)
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
async def warm(model: str, keep_alive: str = "5m", by: str = "unknown",
               force: bool = False) -> dict:
    """Load/pin a model into VRAM for ``keep_alive`` (e.g. ``"5m"``, ``"1h"``).

    Refuses when active reservations leave no room for the model — pass
    ``force=True`` to override. ``by`` records the requester in the audit log.
    """
    return await _in_thread(_warm_impl, model, keep_alive, by, force)


# ── claim tools ──────────────────────────────────────────────────────────────
# The ledger can raise on genuinely-contended/odd filesystem states
# (TimeoutError from a live lock held past the wait window; OSError if the
# Windows sharing-violation retry in claims._save is exhausted). Those are
# operational outcomes, not bugs — surface them as structured {ok: false}
# responses instead of raw tracebacks.

def _ledger_call(verb: str, fn, *args) -> dict:
    """Run a ledger write with the shared failure policy in one place.

    Returns ``{"result": <fn's return>}`` on success, or ``{"error": {ok:
    false, summary}}`` on an OPERATIONAL failure (live-lock timeout,
    exhausted Windows sharing-violation retries) so every claim tool
    degrades identically instead of surfacing a raw traceback.

    ValueError joins that list: the ledger rejects arguments it must not
    persist (``reserve(gb=-8)``), and an MCP caller deserves the same readable
    ``{ok: false, summary}`` for a bad argument as for a busy lock."""
    try:
        return {"result": fn(*args)}
    except (TimeoutError, OSError, ValueError) as e:
        return {"error": {"ok": False, "summary": f"{verb} failed: {e}"}}


def _claim_impl(model: str, owner: str, purpose: str, ttl_seconds: int) -> dict:
    outcome = _ledger_call("Claim", _claims.claim, model, owner, purpose, ttl_seconds)
    if "error" in outcome:
        return outcome["error"]
    result = outcome["result"]
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


def _reserve_impl(gb: float, owner: str, purpose: str, ttl_seconds: int,
                  pid: Optional[int]) -> dict:
    # _ledger_call applies fn(*args) positionally, but reserve's `pid` is
    # keyword-only — so hand it a zero-arg lambda instead of the bare function.
    outcome = _ledger_call(
        "Reserve",
        lambda: _claims.reserve(gb, owner, purpose, ttl_seconds, pid=pid),
    )
    if "error" in outcome:
        return outcome["error"]
    result = outcome["result"]
    result["ok"] = True
    result["summary"] = (
        f"Reserved {gb} GB for {owner} ({purpose}), expires "
        f"{result['expires_at']}. Other sessions' warm() calls will be refused "
        "when this reservation leaves no headroom."
    )
    return result


@mcp.tool()
async def reserve(gb: float, owner: str, purpose: str, ttl_seconds: int = 3600,
                  pid: Optional[int] = None) -> dict:
    """Reserve ``gb`` GB of VRAM — a claim on capacity, not on a named model.

    Use this for non-Ollama GPU work (a training run, a diffusion job) so other
    sessions can see the VRAM is spoken for. ``pid`` is advisory. Reservations
    expire by TTL like claims, so a crashed session never leaves one stuck.

    COOPERATIVE: this gates vram-mcp's own ``warm()``, but vram-mcp cannot
    intercept an Ollama auto-load triggered by a direct /api/generate call
    from another process.
    """
    return await _in_thread(_reserve_impl, gb, owner, purpose, ttl_seconds, pid)


def _renew_impl(claim_id: str, ttl_seconds: Optional[int]) -> dict:
    outcome = _ledger_call("Renew", _claims.renew, claim_id, ttl_seconds)
    if "error" in outcome:
        return outcome["error"]
    result = outcome["result"]
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
    outcome = _ledger_call("Release", _claims.release, claim_id)
    if "error" in outcome:
        return outcome["error"]
    result = outcome["result"]
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


# ── audit trail ──────────────────────────────────────────────────────────────

def _history_impl(model, type_, limit, since) -> dict:
    events = _audit.read_events(model=model, type=type_, limit=limit,
                                since=since, path=_audit.DEFAULT_EVENTS_PATH)
    return {"events": events, "summary": f"{len(events)} event(s)."}


@mcp.tool()
async def history(model: str | None = None, type: str | None = None,
                  limit: int = 50, since: str | None = None) -> dict:
    """The VRAM audit trail, newest first: who ran unload/ensure_free/warm, and
    which models/processes appeared or disappeared (with a best-effort cause).
    Filter by ``model``, ``type`` (action|disappeared|appeared), ``limit``, or an
    ISO ``since`` floor. Answers 'what happened to model X?'."""
    return await _in_thread(_history_impl, model, type, limit, since)


def _trend_impl(hours: float) -> dict:
    since = iso(datetime.now(timezone.utc) - timedelta(hours=hours))
    rows = _audit.read_events(type="sample", limit=10_000, since=since)
    rows.reverse()  # read_events is newest-first; the summarizer needs oldest-first
    summary = _audit.summarize_samples(rows)
    if summary["count"] == 0:
        text = (
            f"No VRAM samples in the last {hours}h. Samples are recorded on "
            f"status calls at most once per {_SAMPLE_SECONDS:g}s "
            "(VRAM_MCP_SAMPLE_SECONDS), and are disabled by VRAM_MCP_AUDIT=0."
        )
    else:
        # "now unknown" rather than a stale number: latest_free_mb describes the
        # NEWEST sample, which may carry no reading at all.
        latest = summary["latest_free_mb"]
        now_text = "unknown" if latest is None else f"{latest} MB"
        text = (
            f"{summary['count']} sample(s) over {hours}h: free VRAM is "
            f"{summary['direction']} (min {summary['min_free_mb']} MB, "
            f"max {summary['max_free_mb']} MB, now {now_text}); "
            f"{summary['thrashing_samples']} sample(s) showed VRAM spilling "
            "to system RAM."
        )
    # The summary above already covers EVERY row; only the raw rows are capped,
    # and the caller is told so rather than silently handed a subset.
    truncated = len(rows) > _TREND_SAMPLE_CAP
    if truncated:
        text += (
            f" Showing the {_TREND_SAMPLE_CAP} most recent raw samples of "
            f"{len(rows)} (the figures above cover all of them)."
        )
    return {**summary, "hours": hours, "samples": rows[-_TREND_SAMPLE_CAP:],
            "samples_truncated": truncated, "summary": text}


@mcp.tool()
async def trend(hours: float = 1.0) -> dict:
    """Free-VRAM trend over the last ``hours``, from the sampled audit log.

    Answers "was this a gradual erosion or a sudden spike?" — the question a
    point-in-time ``vram_status()`` cannot. Returns direction, min/max/latest
    free MB (``latest`` is ``null`` when the newest sample carries no reading),
    how many samples showed driver spill, and the raw samples.

    ``samples`` holds at most the 200 most recent rows so a long window can't
    flood the caller's context; ``samples_truncated`` says whether older rows
    were dropped. Every summary figure is computed over the FULL window either
    way.
    """
    return await _in_thread(_trend_impl, hours)


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
