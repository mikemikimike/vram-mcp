# Pressure, Trend, and Reservations — Design

**Date:** 2026-07-19
**Status:** Approved

## Origin

Field feedback from a separate ML-project session that hit a VRAM wedge and
lost ~20 minutes to a phantom diagnosis. It ranked five gaps. Assessment
against the shipped code (`1e757f7`):

| # | Ask | Verdict |
| --- | --- | --- |
| 1 | Per-process VRAM attribution | **Already shipped** (`e903b93`/`0afa43e`/`b6e5851`). The reporting session was running a subprocess started before that commit. No work needed. |
| 2 | `register_process()` for non-Ollama consumers | **Mostly obsoleted by #1** — attribution now names and sizes them with no registration. The surviving half is *intent* (expected size, priority), which folds into #3. |
| 3 | Real admission control | **Real gap**, with a scope correction (below). |
| 4 | Threshold alerting | **Real gap**, with a protocol correction (below). |
| 5 | Pressure / thrash signal | **Real gap, and nearly free** — same perf-counter sample we already pay for. |

This spec covers 5, 4, and 3+2. #1 needs no code.

## Two corrections to the original asks

**Admission control cannot cover Ollama auto-loads.** Those happen when a
session calls `/api/generate` directly; vram-mcp is not in that path and has no
hook into it. What vram-mcp *can* do is gate its own `warm()`, expose
reservations so cooperating sessions check before loading, and report reserved
VRAM in `ensure_free`. This is **cooperative** admission control. The spec does
not claim enforcement it cannot deliver.

**MCP has no server-initiated messages.** Nothing can push an alert into an
idle session. So #4 ships as a queryable trend, not a `watch()` callback. It
still answers "has free VRAM been eroding?" — but only when something asks.

## Feature A — Pressure signal

### Two distinct failure modes, currently conflated

- **Deliberate CPU offload** — Ollama decides up front to put some layers on
  the CPU backend. Detected today via `size_vram < size` (`offloaded_to_cpu`).
  Slower, but a considered choice.
- **Driver-forced spill** — WDDM demand-pages VRAM out to system RAM under
  pressure. This is the 7×-slowdown mode. **Not detected today at all.**

Verified live: `qwen3:32b` showed `offloaded_to_cpu: true` (20,233 of 27,802 MB
resident) while `Non Local Usage` was ~0 — proving the two are independent and
that reporting only the first is insufficient.

### Mechanism

`Get-Counter -ListSet 'GPU Process Memory'` exposes five counters. We already
sample `Dedicated Usage`. Verified live as nonzero and per-PID:

```
\GPU Process Memory(*)\Dedicated Usage    ← already used
\GPU Process Memory(*)\Shared Usage       ← add
\GPU Process Memory(*)\Non Local Usage    ← add (the spill signal)
```

All three come from **one** `Get-Counter` call with three paths, so the cost
stays at the single ~1 s sample already being paid. Samples are discriminated
by matching the counter `Path`.

### Shape

`process_table()` rows gain `shared_mb` and `non_local_mb` (both `int | None`).

New pure function `core.pressure(gpus, loaded, other_processes)` returns:

```python
{
  "state": "ok" | "tight" | "degraded" | "thrashing",
  "free_mb": int | None,
  "non_local_mb": int,            # summed driver-spilled VRAM
  "spilling": bool,               # non_local_mb >= SPILL_THRESHOLD_MB
  "offloaded_models": [str],      # models with offloaded_to_cpu
  "detail": str,                  # one human-readable sentence
}
```

State precedence (first match wins):

1. `thrashing` — `spilling` is True. Driver is paging VRAM; expect severe slowdown.
2. `degraded` — one or more models are offloaded to CPU. Slower by choice.
3. `tight` — `free_mb` is known and below `TIGHT_MB` (1024).
4. `ok` — none of the above.

`SPILL_THRESHOLD_MB = 256` — below that, non-local usage is normal desktop noise.

Surfaced as a top-level `pressure` key on `vram_status()` and folded into its
`summary` string so it is visible without inspecting the payload.

## Feature B — Trend samples

A `type="sample"` event appended to the existing `events.jsonl`:

```python
{"ts", "type": "sample", "free_mb", "used_mb", "total_mb",
 "non_local_mb", "state", "loaded_count"}
```

**Throttled** to at most one sample per `SAMPLE_INTERVAL_SECONDS` (default 60)
across all sessions, via a small `~/.cache/vram-mcp/last_sample.json` holding
the last sample timestamp, read and written under the shared lock. Without the
throttle, a busy multi-session period would flood the 5000-event cap and evict
the action/disappearance events that carry the real diagnostic value.

A dedicated state file (rather than reusing `last_seen.json`) keeps the
sampling and detection concerns separable; the extra IO is a sub-millisecond
JSON read against a ~1 s perf-counter sample already in flight.

New tool `trend(hours=1)` summarizes the samples in the window: sample count,
min/max/latest free MB, direction (`falling`/`rising`/`flat` by comparing the
first and last thirds), and how many samples were in a spilling state. This is
what answers "was this a gradual erosion or a sudden spike?".

Samples are gated by `VRAM_MCP_AUDIT` like the rest of passive detection.

## Feature C — Reservations

### Model

A reservation is a claim on **GB of VRAM**, not on a named model. It lives in
the same `claims.json` ledger with the same TTL/crash-safety semantics, tagged
`kind: "reservation"`:

```python
{"claim_id", "kind": "reservation", "model": None, "gb": float,
 "pid": int | None, "owner", "purpose",
 "claimed_at", "renewed_at", "ttl_seconds", "expires_at"}
```

Existing model claims are tagged `kind: "model"`. Records written by older
versions have no `kind`; they are treated as `"model"` so the ledger stays
backward compatible.

`pid` is optional and advisory — it records *which* process the reservation is
for, absorbing the useful half of the original `register_process()` ask.

Existing behavior is unaffected: `core.Snapshot.claims_for(name)` matches on
`model == name`, and a reservation's `model` is `None`, so reservations never
protect or shadow a model.

### Admission control

`core.reserved_mb(all_claims)` sums active reservations. Then:

- **`warm(model, ..., force=False)`** — computes
  `headroom = free_mb - reserved_mb`. Refuses when the model will not fit,
  using the model's size from `/api/tags` when known; when the size is unknown
  it refuses only if `headroom <= 0`. `force=True` overrides. Refusals are
  logged as `outcome="refused"` with the reservation detail.
- **`ensure_free(gb, ...)`** — unchanged in behavior, but its result gains
  `reserved_mb` and the summary notes when freed VRAM is already spoken for.
  Unloading to hit a target that reservations have already claimed is a
  footgun worth naming rather than silently permitting.

`OllamaClient` gains `tags()` (`GET /api/tags`) to look up on-disk model size.
Disk size is an approximation of VRAM need, not an equality — it is used only
to make a refusal decision, and the refusal is always overridable.

### New tools

- `reserve(gb, owner, purpose, ttl_seconds=3600, pid=None)` → `{claim_id, expires_at}`
- `renew()` / `release()` / `list_claims()` work on reservations unchanged
  (they operate by `claim_id`).

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VRAM_MCP_SAMPLE_SECONDS` | `60` | Minimum seconds between trend samples. |
| `VRAM_MCP_SPILL_MB` | `256` | Non-local VRAM above which `spilling` is True. |

Existing `VRAM_MCP_AUDIT`, `VRAM_MCP_MEANINGFUL_MB`, `VRAM_MCP_EVENT_CAP`
are unchanged. `VRAM_MCP_AUDIT=0` disables the perf-counter sample, so it
disables pressure detail and trend sampling too — documented, not silent.

## Non-goals

- Killing or throttling non-Ollama processes. vram-mcp reports them; it does
  not manage them.
- Push alerts. Not expressible in MCP.
- Intercepting Ollama auto-loads. Not in vram-mcp's path.
- Priority-based preemption between reservations. Reservations are advisory
  and first-come; a priority scheme needs a real arbitration policy and is out
  of scope until there is evidence it is needed.

## Testing

Every module stays pure and injectable, matching the existing convention: no
`mcp` import outside `server.py`, all readers/clocks/paths injected, and the
full suite runnable with no GPU, no Ollama, and no `mcp` package.

- `procinfo` — a fake three-counter PowerShell output parses to the right
  `shared_mb`/`non_local_mb`; a pid present in one counter but not another;
  a cmdline containing `|`.
- `core.pressure` — each state, precedence between them, `free_mb is None`,
  all-`None` non-local values.
- `audit` sampling — throttle honored, first sample always written, corrupt
  state file treated as "never sampled", `summarize_samples` direction logic.
- `claims` — reservation round-trip, TTL expiry, `kind` back-compat for
  records with no `kind`, reservations invisible to `claims_for`.
- `core.reserved_mb` — active only, malformed records skipped.
- `warm` admission — fits/does-not-fit/unknown-size/force paths.
