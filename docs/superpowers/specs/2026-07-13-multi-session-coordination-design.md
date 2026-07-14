# Multi-Session VRAM Coordination — Design Spec

**Date:** 2026-07-13
**Status:** Approved (design decisions locked); pending spec self-review → writing-plans.

## Goal

vram-mcp's whole purpose is letting **separate, independent LLM sessions** (each running its
own `vram-mcp` subprocess, no shared server today) coordinate use of one GPU's VRAM. Today it
can only report Ollama's own model list + raw VRAM numbers — it has **no way to tell one
session what another session is actually doing** with a resident model, and no way to detect
non-Ollama VRAM consumers at all. This spec adds:

1. **Attribution** — a session declares *why* it's using a model (a "claim"), so other
   sessions/humans see who's using what and why before deciding to evict it.
2. **Busy detection** — a best-effort, real (not just inferred) signal for whether a specific
   model is actively computing right now, via NVML.
3. **Full VRAM visibility** — every process holding VRAM, not just Ollama models (a game, a
   training job, another inference server), via the same NVML calls.

## Locked decisions

- **New dependency:** `nvidia-ml-py` (NVIDIA's official, BSD-licensed, dependency-free NVML
  bindings). Confirmed acceptable — replaces the earlier idea of shelling out to
  `nvidia-smi pmon` (fragile CLI-text parsing) with a stable, typed API.
- **Attribution is self-reported, not inferred.** Ollama's API has no concept of "caller" or
  "purpose" — this can only ever come from the session declaring it via a new `claim()` tool.
- **Busy detection is NVML-based and windowed**, not point-in-time. `nvmlDeviceGetProcessUtilization`
  returns samples across NVML's short internal buffer (not a single instant), so brief idle
  gaps between generated tokens don't falsely read as "idle" the way a single `pmon -c 1`
  snapshot could.
- **PID→model-name correlation is isolated and explicitly fragile.** Ollama spawns one
  `llama-server` subprocess per loaded model but exposes no PID in its own API — the only
  correlation path is reading the runner's command-line model-blob hash and matching it
  against Ollama's on-disk manifest files. Undocumented, could break on an Ollama version
  change. Contained in its own module (`ollama_correlate.py`) so a break there degrades one
  signal (`busy` → `null`) rather than the whole tool.
- **All-process VRAM visibility is in scope now** (not deferred) — the same NVML process-list
  calls needed for busy-detection PID correlation give this almost for free.
- **A model is protected from default eviction if EITHER it has an active claim OR its
  best-effort `busy` signal is `true`** — not claim-status alone. An uncooperative caller that
  never calls `claim()` still shouldn't be able to make a real in-flight generation trivially
  interruptible. `force=true` overrides both.
- **Claims expire via TTL, not explicit release only.** A session can crash/be killed without
  cleanup (observed behavior in practice); relying solely on an explicit `release()` would
  leave stale claims stuck forever. An un-renewed claim simply stops being active once its
  `expires_at` passes.

## Architecture

Three new modules alongside the existing `gpu.py` / `ollama.py` / `core.py` (all pure, no
`mcp` import, injectable for tests — matching the existing pattern):

- **`nvml.py`** — wraps `nvidia-ml-py`. Provides: accurate free/used/reserved VRAM
  (`nvmlDeviceGetMemoryInfo_v2`), every VRAM-holding process with PID + size + kind
  (`nvmlDeviceGetComputeRunningProcesses_v3` + `nvmlDeviceGetGraphicsRunningProcesses_v3`),
  windowed per-process busy signal (`nvmlDeviceGetProcessUtilization`), and stable per-GPU
  identity (`nvmlDeviceGetUUID`). Degrades to "unavailable" (`None`/`[]`) on any NVML init/call
  failure — no NVIDIA driver, package not installed, non-NVIDIA GPU, WDDM field gaps. `gpu.py`'s
  existing `nvidia-smi` parsing remains the fallback path when NVML itself is unavailable.
- **`ollama_correlate.py`** — maps a `llama-server` subprocess PID to the Ollama model tag it's
  serving (command-line blob-hash → manifest lookup). Returns `None` on any failure (wrong
  Ollama version, manifest not found, ambiguous match) rather than raising — callers always
  treat "no correlation" as "unknown," never a wrong guess.
- **`claims.py`** — the shared attribution ledger. `claim(model, owner, purpose, ttl_seconds)`,
  `renew(claim_id)`, `release(claim_id)`, `list_claims(model=None)`. Backed by a JSON file at
  `~/.cache/vram-mcp/claims.json` (since every session runs its own independent `vram-mcp`
  process — there is no shared long-lived server to hold this in memory). Writes are
  atomic (temp file + rename) and serialized via a sibling lock file so concurrent sessions
  can't corrupt each other's writes.
- **`core.py`** (existing, expanded) — stays the sole orchestrator: for each Ollama `/api/ps`
  entry it now also calls `ollama_correlate` for a PID, then `nvml.process_utilization(pid)` for
  the busy signal, and `claims.list_claims(model=name)` for attribution, merging all three into
  the per-model dict `server.py` returns. `nvml.py` and `ollama_correlate.py` never call each
  other or `claims.py` directly — `core.py` is the only module that combines signals across
  them, matching its existing role combining `gpu_status()` + `ollama.ps()`.

## On-disk claim schema

```json
{
  "claims": [
    {
      "claim_id": "b3f1c2a4-...",
      "model": "llama3.2",
      "owner": "retro-repo",
      "purpose": "narration enrichment",
      "claimed_at": "2026-07-13T18:00:00Z",
      "renewed_at": "2026-07-13T18:05:00Z",
      "ttl_seconds": 3600,
      "expires_at": "2026-07-13T19:05:00Z"
    }
  ]
}
```

A claim is **active** while `now < expires_at`. Multiple sessions may each hold their own
active claim on the same model (e.g. two projects sharing `llama3.2`) — a model counts as
"claimed" for eviction purposes if *any* active claim exists on it. `renew()` extends
`expires_at` to `now + ttl_seconds`; it does not require the original `ttl_seconds` to match.
All timestamps are UTC ISO 8601 (`...Z`), matching Ollama's own `expires_at` format. `owner`
is a free-form human-readable label the caller chooses (e.g. a project/repo name or session
name) — not a machine-parsed session ID; it exists purely so a human or another agent reading
`vram_status()` can understand who's using the model, not to uniquely key anything.

## Tool surface changes

**New tools:**
- `claim(model: str, owner: str, purpose: str, ttl_seconds: int = 3600) -> {claim_id, expires_at}`
- `renew(claim_id: str) -> {ok, expires_at}` — `ok=False` if the claim doesn't exist / already expired.
- `release(claim_id: str) -> {ok}`
- `list_claims(model: str | None = None) -> {claims: [...]}` — all active claims, optionally filtered to one model.

**Changed tools:**
- `vram_status()` / `list_loaded()` — each Ollama model entry gains `claim` (the matching
  active claim summary, or `null`), `busy` (`true`/`false`/`null`), `offloaded_to_cpu` +
  `total_size_mb` (derived from `size` vs `size_vram`, previously dropped — see prior finding).
  A new top-level `other_processes` list reports every NVML-visible VRAM holder that isn't an
  Ollama model: `{pid, name, size_mb, kind: "compute" | "graphics"}`.
- `unload(model, force: bool = False)` / `ensure_free(gb, force: bool = False)` — protected
  models (active claim OR `busy == true`) are skipped by default; `ensure_free`'s response
  reports which protected models it declined to evict (with their claim/busy detail) even when
  the VRAM target wasn't fully reached. `force=True` bypasses protection on both.

## Error handling

- NVML unavailable at all → every NVML-derived field is `null`/`[]`; `gpu.py`'s `nvidia-smi`
  path still supplies basic total/used/free so the tool keeps working, degraded.
- PID correlation fails for a given model → that model's `busy` is `null`; never blocks the
  rest of `vram_status()`.
- Claim-ledger lock contention → short retry, then the `claim`/`renew`/`release` call fails
  explicitly (not a silent corruption or a hang).
- Crashed session's claim → no cleanup needed; it simply stops being active once `expires_at`
  passes.
- Windows WDDM → `reserved` memory and some process VRAM sizes may be unavailable under NVML;
  those specific fields fall back to the existing `nvidia-smi` numbers.

## Testing

Same philosophy as the existing suite: `nvml.py` and `ollama_correlate.py` stay pure and fully
unit-testable with injected fakes — no real GPU, driver, or Ollama daemon required. `claims.py`
is tested against a real temp directory (atomicity/locking is the actual thing under test, so a
mock would test nothing meaningful). Required cases: claim/renew/release lifecycle + expiry
math; concurrent-write safety; `ensure_free` skipping protected models and reporting what it
declined; `busy=null` propagation on correlation/NVML failure; WDDM fallback for memory info;
`other_processes` classification (compute vs graphics vs unknown).

## Out of scope (this spec)

- NVML accounting mode (root/admin required, doesn't track VRAM bytes, low fit — not adopted).
- Any change to how calling sessions actually issue their real Ollama inference requests (no
  per-call wrapping) — busy detection is entirely NVML-side, zero integration cost for callers
  beyond calling `claim()` once at load time.
- Non-NVIDIA GPU backends (AMD/Intel) — already listed as a future roadmap item in the README,
  unaffected by this spec.
