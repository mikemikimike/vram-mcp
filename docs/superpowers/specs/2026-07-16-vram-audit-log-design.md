# VRAM Audit Log + Meaningful-Model Detection — Design Spec

**Date:** 2026-07-16
**Status:** Approved (design decisions locked); pending spec review → writing-plans.

## Goal

vram-mcp can currently tell you the *present* state of the GPU, but keeps **no
history**: when a model leaves VRAM, there is no record of who removed it or when.
It is also blind to *non-Ollama* GPU processes by name/size (on Windows/WDDM every
`other_processes` entry shows `size_mb: null`), so a large PyTorch job — e.g. a
FLUX LoRA training run holding 14.5 GB — appears only as an anonymous PID.

This spec adds:

1. **Meaningful-model detection** — resolve each process's *dedicated VRAM* and
   name, universally (NVML where available, a Windows perf-counter fallback where
   NVML returns null), so a real model (≥ a size threshold) is distinguishable
   from browser/shell GPU noise **without a name allowlist**.
2. **An append-only audit log** — every mutating tool call, plus best-effort
   detection of *disappearances/appearances* of meaningful VRAM holders, recorded
   as typed events with honest cause attribution.
3. **A `history` tool** — so "what happened to model X?" is answerable.

## Key discovery (validated live on the dev machine)

Per-process **dedicated** VRAM *is* reliably available on Windows/WDDM via the
`\GPU Process Memory(*)\Dedicated Usage` performance counter — the same source
Task Manager uses — provided instances are summed per PID. Verified:

- `python` (a FLUX LoRA training run) → 14,492 MB; `UnrealEditor` → 2,386 MB;
  `dwm` → 1,756 MB; browser/shell → tens of MB.
- Per-process dedicated sum (19,592 MB) ≈ `nvidia-smi` used (18,546 MB), within
  ~5% — the counter accounts for essentially all VRAM, so it is trustworthy.
- Cost: one `Get-Counter` sample ≈ **1 s** (full enumeration). This is the reason
  the perf-counter path is a *fallback*, used only on the status tools and only
  when NVML sizes are null — never on the fast tools.

An earlier "dwm = 47 GB" reading was a conflated `Shared`/`Total Committed`
instance, not `Dedicated Usage`; the dedicated counter is the correct one.

## Locked decisions

- **The disappearance filter is SIZE, not names.** A "meaningful holder" is any
  process with dedicated VRAM ≥ `VRAM_MCP_MEANINGFUL_MB` (default 512). Principled
  and universal; no name allowlist to maintain.
- **Size resolution is NVML-first, perf-counter-fallback.** NVML
  `usedGpuMemory` when non-null (Linux/TCC); the `Dedicated Usage` perf counter
  (summed per PID) only when NVML returns null (Windows/WDDM). The ~1 s
  perf-counter call runs **only inside the once-per-call snapshot on the status
  tools** (`vram_status`/`list_loaded`), never on `claim`/`renew`/`unload`.
- **One unified, typed, append-only event log** (`~/.cache/vram-mcp/events.jsonl`)
  — not two files — so `history` returns a single merged timeline.
- **Disappearance detection is diff-based, no daemon.** A `last_seen.json`
  (overwritten each status call) holds the current meaningful-holder set; each
  status call, under one lock, reads last-seen → resolves current → diffs →
  appends events → rewrites last-seen. The lock serializes concurrent sessions so
  a disappearance is logged **once**, not once per polling session.
- **Attribution is honest, never invented.** Self-caused evictions are attributed
  to the recorded action; everything else is `external` (Ollama models) or
  `unattributed` (non-Ollama process exits), with a plain-language `detail`.
- **Actions record a self-reported actor.** Eviction tools gain an optional `by`
  (owner) param, same philosophy as claims; defaults to `"unknown"`.
- **The audit may never break a tool call.** Any log/resolver failure is swallowed;
  perf-counter absence degrades to null sizes (those processes simply aren't
  disappearance-tracked), never raises.
- **Bounded, crash-safe storage.** Event log capped at `VRAM_MCP_EVENT_CAP`
  (default 5000), pruned on append; atomic writes + the shared file lock already
  hardened for claims (generalized into `_util`, not re-implemented).
- **Opt-out.** `VRAM_MCP_AUDIT=0` disables detection + the perf-counter cost.

## Architecture

New/changed modules (all pure, injectable, no `mcp` import — matching the codebase):

- **`vram_mcp/procinfo.py`** (new) — `process_table(*, nvml_processes, perf_reader,
  proc_lister) -> list[dict]`: merges NVML process/size data with a Windows
  perf-counter size fallback and PID→name/cmdline labeling. Returns
  `[{pid, name, cmdline, size_mb, kind}]`. All external calls injected.
- **`vram_mcp/audit.py`** (new) — the event log + diff detector.
  `log_action(event, *, path, now_fn, cap)`, `detect_and_log(current_holders, *,
  last_seen_path, log_path, now_fn)`, `read_events(model, type, limit, since, *,
  path)`. Uses the shared lock/atomic-write helper.
  - **Holder identity** (what makes a holder "the same" across two snapshots):
    the key is `"ollama:<model_name>"` for Ollama models and `"process:<pid>"`
    for non-Ollama processes; each stored entry also carries the display fields
    (`name`, `cmdline`, `size_mb`) so a `disappeared` event can name what left.
    PID reuse within the sub-second-to-seconds gap between status calls is
    negligible; a reused PID at worst mislabels one event's display text, never
    corrupts the log.
  - **Recent-action lookup for attribution** is derived by reading the tail of
    `events.jsonl` for `action` events within the last 10 s (same file, same
    read) — no separate store.
- **`vram_mcp/_util.py`** (existing) — the `claims._locked` + atomic-write logic
  is generalized here and reused by both `claims.py` and `audit.py`.
- **`vram_mcp/nvml.py`** (existing) — unchanged except `nvml_processes` already
  returns `size_mb` (null on WDDM); `procinfo` fills the gap.
- **`vram_mcp/core.py`** (existing) — `combined_status` gains a `procinfo_fn` so
  `other_processes` carries real `size_mb`/`name`/`cmdline`; a hook to run
  `audit.detect_and_log` on the resolved holder set.
- **`vram_mcp/server.py`** (existing) — status tools wire `procinfo` + detection;
  mutating tools log an `action` event and gain the optional `by` param; new
  `history` tool.

## Windows perf-counter reader

A small injected `perf_reader() -> dict[int, int]` (PID → dedicated MB), summing
`\GPU Process Memory(*)\Dedicated Usage` instances per PID via one PowerShell
`Get-Counter` call. Returns `{}` on any failure (non-Windows, counter absent,
timeout). Invoked only when NVML sizes are null and audit is enabled.

PID→name/cmdline reuses the process-listing machinery already built for
`ollama_correlate` (wmic → PowerShell CIM fallback), generalized to list *any*
process, not just `llama-server`.

## Event schema (`events.jsonl`, one JSON object per line)

```json
{ "ts": "2026-07-16T18:05:03Z",
  "type": "action | disappeared | appeared",
  "kind": "ollama | process",
  "target": "qwen3:8b",
  "size_mb": 14492,
  "action": "unload | ensure_free | warm | claim | renew | release",
  "actor": "retro-repo | unknown",
  "force": true,
  "outcome": "ok | refused | failed",
  "cause": "self_action | external | unattributed",
  "detail": "human-readable summary" }
```

Field presence: `action`/`actor`/`force`/`outcome` for `type=action`;
`cause`/`detail` for `type=disappeared`; `size_mb` for process events (may be null).

## Attribution rules (for `disappeared`)

- **Ollama model** gone AND a matching `unload`/`ensure_free` action was logged
  within the last 10 s → `cause: self_action`, `actor` copied from that action;
  `detail`: "unloaded by <actor> via <action>".
- **Ollama model** gone, no matching recent action → `cause: external`;
  `detail`: "no vram-mcp action recorded — Ollama idle-expiry, memory-pressure
  eviction, or an external unload." (vram-mcp does not parse Ollama's server.log
  to disambiguate these; the user can consult it. Out of scope.)
- **Non-Ollama process** gone → `cause: unattributed`; `detail`: "process exited
  or was killed; vram-mcp cannot observe the cause."

## Data flow

`vram_status` → resolve GPUs + Ollama models + `procinfo` (sized, named) → build
the meaningful-holder set (Ollama models ∪ non-Ollama processes ≥ threshold) →
`audit.detect_and_log` diffs it against `last_seen.json` under the lock, appends
disappeared/appeared events with attribution, rewrites last-seen → tool returns
enriched status. A mutating tool (`unload` etc.) additionally appends an `action`
event before/after acting. `history(...)` reads `events.jsonl`.

## `history` MCP tool

`history(model: str | None = None, type: str | None = None, limit: int = 50,
since: str | None = None) -> {"events": [...], "summary": str}` — recent events
newest-first, optionally filtered by target model, event type, or an ISO
timestamp floor.

## Configuration (env)

- `VRAM_MCP_MEANINGFUL_MB` — meaningful-holder size threshold (default `512`).
- `VRAM_MCP_EVENT_CAP` — max retained events (default `5000`).
- `VRAM_MCP_AUDIT` — `0` disables detection + the perf-counter cost (default on).

## Error handling

- Perf counter unavailable / non-Windows without NVML sizes → process `size_mb`
  stays null; such processes are excluded from disappearance tracking (can't
  threshold), noted once in the resolver, never raised.
- Event-log or last-seen write failure → swallowed (best-effort; a tool call
  never fails because the audit couldn't write).
- `last_seen.json` missing/corrupt → treated as empty: current holders become the
  baseline, so a first run (or a reset) never emits a false "everything
  disappeared" storm.
- Concurrent sessions → the detect-and-log critical section holds the shared lock;
  the first session to observe a disappearance logs it and removes it from
  last-seen, so peers don't double-log.

## Testing

Pure modules, everything injected (perf-counter reader, process lister, NVML,
`now_fn`, `tmp_path` for both files — real I/O since atomicity/locking is under
test, matching `test_claims.py`). Required cases:

- `procinfo`: NVML size used when present; perf-counter fallback when NVML null;
  PID→name/cmdline labeling; perf-counter failure → null sizes, no raise.
- threshold filter: a ≥-threshold process is a meaningful holder; a sub-threshold
  one is not.
- diff detector: present→absent emits exactly one `disappeared`; absent→present
  emits `appeared`; unchanged emits nothing.
- attribution: `self_action` (matching recent action), `external` (Ollama, no
  action), `unattributed` (non-Ollama process) — all three paths.
- concurrency: two diffs against the same last-seen under the lock log the
  disappearance once.
- retention: appending past the cap rewrites to the last N.
- first-run/reset: empty/corrupt last-seen → baseline set, zero disappeared events.
- audit disabled (`VRAM_MCP_AUDIT=0`): no perf-counter call, no events written.
- `history` filtering by model/type/since/limit.

## Out of scope

- Parsing Ollama's `server.log` to classify an `external` Ollama eviction as
  idle-expiry vs memory-pressure vs explicit unload (possible future enhancement).
- A real-time background monitor / daemon (rejected: breaks the crash-only,
  on-demand design; diff-on-status-call is the chosen middle path).
- Killing or preventing eviction of non-Ollama processes (vram-mcp observes them
  but does not manage them).

## Future note — `advise()` v2 (separate follow-up, NOT this spec)

The current `advise()` recommends only `OLLAMA_MAX_LOADED_MODELS=1` and a finite
`OLLAMA_KEEP_ALIVE`. Ollama 0.32.0's env surface (from `ollama serve --help`)
exposes several higher-value VRAM levers `advise()` could suggest based on the
state it already computes (esp. `offloaded_to_cpu` and low free VRAM):

- `OLLAMA_KV_CACHE_TYPE=q8_0` — biggest lever; ~halves KV-cache VRAM, can make an
  offloaded model fit fully on-GPU. **Requires `OLLAMA_FLASH_ATTENTION=1`**
  (verify this coupling at build time) — advise the pair together.
- `OLLAMA_FLASH_ATTENTION=1` — cuts KV-cache memory + faster.
- `OLLAMA_CONTEXT_LENGTH` — KV cache scales with context; an oversized default can
  spill a model to CPU. Advise lowering when a model is offloaded.
- `OLLAMA_GPU_OVERHEAD` / `LLAMA_ARG_FIT_TARGET` — reserve a VRAM margin so
  Ollama's scheduler leaves room for a **non-Ollama** job (e.g. a FLUX training
  run). This is the Ollama-side complement to vram-mcp's whole coordination
  purpose — the most on-theme addition.
- `OLLAMA_NUM_PARALLEL` — each parallel slot multiplies KV cache; advise lowering
  on a tight GPU.

Also flagged for later: 0.32.0 added **experimental image generation**
(`ollama run --width/--height/--steps/--seed`). If diffusion models eventually run
*through* Ollama, this spec's "Ollama = LLM-only; FLUX is a foreign process"
assumption changes and the resolver/attribution split may need revisiting.
Not actionable now; revisit when it lands.
