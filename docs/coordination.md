# Coordination and diagnostics

[Back to the README](../README.md)

## Shared state

Every connected client starts its own stdio server. Processes running under the
same OS user coordinate through `~/.cache/vram-mcp/`:

| File | Contents |
| --- | --- |
| `claims.json` | Model claims and capacity reservations, with owners, purposes, and expiry times. |
| `events.jsonl` | Bounded audit history of actions, observed changes, and memory samples. |
| `last_seen.json` | Previous set of observed memory holders. |
| `last_sample.json` | Shared sampling throttle state. |

Ledger writes use file locks and atomic replacement. Owners are descriptive
labels, not authenticated identities. Separate home directories have separate
ledgers, even when their processes use the same physical GPU.

## Model claims

Use `claim(model, owner, purpose)` before relying on a model. Claims default to
3,600 seconds and multiple sessions may claim the same model independently.
A claim records use; it neither loads the model nor extends Ollama's residency.

Use the **exact model name**, including its tag, as reported by Ollama. Claims
are matched by string: claiming `llama3` does not protect `llama3:latest`.
For a model not currently loaded, obtain its installed name with `ollama list`
before claiming and warming it.

Check `ok` and save the returned `claim_id` and `expires_at`. Renew before
expiry; `renew` cannot revive an expired claim. Release when work ends. If a
session exits without cleanup, its claim expires naturally. Claim expiry and
release do not trigger an unload.

## Eviction and reservations

`unload()` and `ensure_free()` protect models with an active claim or
`busy=true`. Activity is a best-effort NVML reading over a recent window;
`busy=null` supplies no protection by itself. Use claims even when activity
readings are available.

`ensure_free(gb=8)` targets 8 × 1,024 MB and checks the **largest free-memory
reading on any single GPU**, rather than pooling memory across cards. It
unloads candidates by decreasing VRAM size, rechecks memory after each attempt,
and reports skipped models in `declined`. It does not select which GPU Ollama
will use. If telemetry is unavailable, it can exhaust unprotected candidates
and still return `ok=false` because the target cannot be verified.

For non-Ollama work, `reserve(gb, owner, purpose, ttl_seconds=3600, pid=None)`
records capacity in the same ledger. `pid` is advisory; it does not monitor
process lifetime or release the reservation automatically. Use the returned
`claim_id` with `renew` and `release`. Unfiltered `list_claims()` includes
reservations; filtering by `model` excludes them.

Reservations do not allocate memory, validate availability, or protect a named
model from eviction. They are reported as `reserved_mb` by `ensure_free()` and
checked by `warm()`. Inspect existing reservations and available memory before
starting a job. The ledger is cooperative and these calls do not form an atomic
allocation transaction; direct Ollama requests bypass it.

### Interpreting a warm result

A normal `warm()` checks free memory minus active reservations against an
approximate model size from Ollama's installed-model list. Its `reason` explains
the admission decision:

| `reason` | Meaning |
| --- | --- |
| `fits` | The estimated size fits the remaining headroom; `size_verified=true`. |
| `no_reservations` | No reserved capacity was found; no model-fit check is required. |
| `free_unknown` | Free memory could not be read; the load is allowed without verifying fit. |
| `size_unknown` | Reservations exist and headroom is positive, but model size is unavailable; the load is allowed without verifying fit. |
| `no_headroom` | Reservations leave no free headroom; the load is refused. |
| `insufficient_headroom` | The estimated model size exceeds remaining headroom; the load is refused. |

`ok` reports the load request's outcome; it does not prove the model is fully
GPU-resident. `size_verified=false` means the `fits` check did not pass or run;
read `reason` to distinguish cases. Even `fits` uses on-disk size as an estimate,
not a guarantee of runtime VRAM needs.

An unreadable ledger is treated as no reservations for warming. A forced warm
skips admission and omits its verdict fields. `force=True` overrides claims and
busy protection for eviction, or reservations for warming; resolve competing
work before using it. Recheck status after a mutation rather than assuming the
requested state was reached.

## Diagnostics

### Pressure and CPU offload

The pressure verdict prioritizes `thrashing`, then `degraded`, then `tight`,
then `ok`. Read it alongside `free_mb`, process readings, and `pressure.detail`.
Unavailable telemetry can produce `ok` without establishing available capacity.

On Windows/WDDM, a runner's non-local memory can include deliberately
CPU-offloaded layers. vram-mcp attributes that memory before judging spill:

- `non_local_mb`: raw total reported by the process table.
- `explained_offload_mb`: memory attributable to Ollama's deliberate CPU offload,
  capped per correlated runner at `total_size_mb - size_vram_mb`.
- `unexplained_spill_mb`: the remainder, including other processes' non-local
  memory and any excess above a runner's explained offload.

Spill is flagged when unexplained memory reaches `VRAM_MCP_SPILL_MB` (default
256 MB) **and exceeds free VRAM**. If free VRAM is unknown, only the floor can
be checked. Treat this as a diagnostic heuristic. A partially CPU-offloaded
model can correctly read `degraded` without evidence of driver paging.

### History

Use MCP tool calls such as `history(model="llama3:latest")` or
`history(type="disappeared", limit=25)`. Results are newest-first; `since` accepts
an ISO timestamp.

Action events record warm/unload outcomes and ensure-free evictions or refusals,
with the caller's `by` label (default `unknown`). An `ensure_free` call that finds
sufficient memory immediately has no eviction action to record.

With auditing enabled, `vram_status()` and `list_loaded()` compare current
holders with the previous snapshot. All Ollama models and non-Ollama processes
using at least `VRAM_MCP_MEANINGFUL_MB` dedicated VRAM qualify. The first snapshot
establishes a baseline; changes between observations can be missed.

Disappearance causes are best-effort attribution:

| Cause | Interpretation |
| --- | --- |
| `self_action` | A recent matching vram-mcp eviction explains the disappearance. |
| `external` | No matching eviction was recorded; possibilities include Ollama expiry, pressure, or an external unload. |
| `unattributed` | A non-Ollama process disappeared; its exit cause is unknown. |

Audit failures do not fail the main tool call, so history is useful evidence,
not a complete record of every GPU event.

### Trends

`trend(hours=1)` summarizes retained samples: direction, min/max/latest free MB,
and samples showing driver spill. Sampling happens only during `vram_status()`
calls that return GPU readings, at most once per `VRAM_MCP_SAMPLE_SECONDS`
across sessions. `list_loaded()` does not collect GPU readings or trend samples.

The response includes at most the 200 most recent raw samples and sets
`samples_truncated` when more exist. Summary figures cover the queried retained
samples, not just the returned raw subset; the implementation queries at most
10,000 samples. Event retention can shorten the available history. An empty
window is not evidence of stable memory, and a null latest reading is unknown.

See [configuration](configuration.md) for sampling, retention, and the effects
of disabling auditing.
