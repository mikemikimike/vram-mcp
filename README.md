# vram-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents **share
one NVIDIA GPU** safely. Built for the reality of several Claude Code / agent
sessions juggling [Ollama](https://ollama.com) models on a single card: before
loading the next model, an agent can see exactly what's holding VRAM, who's
using it and why, whether it's computing *right now* — and free space without
stepping on another session's in-flight work.

- **Claims** — sessions declare *who* is using a model and *why*, in a shared
  crash-safe ledger. TTL-based: a killed session never leaves a stuck claim.
- **Real busy detection** — windowed per-process GPU utilization via NVML,
  with zero changes to how anything calls Ollama.
- **Protected eviction** — `unload`/`ensure_free` refuse to evict a claimed or
  actively-computing model by default; `force=True` when you've decided.
- **Full visibility** — every VRAM-holding process on the GPU, not just Ollama
  models, plus CPU-offload detection (`size_vram < size` = spilled to RAM).
- **Degrades gracefully** — no `nvidia-smi`/NVML/`wmic`? Readings become
  `unknown`/`null`, never wrong; model list / unload / warm keep working.
- NVIDIA + Ollama for now (see [Roadmap](#roadmap)).

## Tools

| Tool | Behavior |
| --- | --- |
| `vram_status()` | Per-GPU VRAM (total/used/free) + loaded Ollama models (with claims, busy signal, CPU-offload) + every other VRAM-holding process + best free MB. |
| `list_loaded()` | The models currently resident in VRAM (name, VRAM MB, expiry, claims, busy). |
| `unload(model, force=False, by="unknown")` | Evict one model from VRAM now (`keep_alive=0`). Refuses if claimed/busy unless `force=True`. `by` records the requester in the audit log. |
| `ensure_free(gb, force=False, by="unknown")` | Unload models largest-first until at least `gb` GB is free, skipping claimed/busy models unless `force=True`. `by` records the requester in the audit log. |
| `warm(model, keep_alive="5m", by="unknown")` | Load/pin a model into VRAM for a duration. `by` records the requester in the audit log. |
| `advise()` | Heuristic suggestions (e.g. `OLLAMA_MAX_LOADED_MODELS=1`, finite `OLLAMA_KEEP_ALIVE`). |
| `claim(model, owner, purpose, ttl_seconds=3600)` | Declare you're using a model, so others see who/why before evicting it. |
| `renew(claim_id, ttl_seconds=None)` | Extend a claim before it expires. |
| `release(claim_id)` | Release a claim early. |
| `list_claims(model=None)` | See active claims (all models, or one). |
| `history(model=None, type=None, limit=50, since=None)` | The audit trail, newest first: who ran `unload`/`ensure_free`/`warm`, and which models/processes appeared or disappeared (with a best-effort cause). |

## Requirements

- **Ollama** running locally (or reachable via `OLLAMA_BASE_URL`).
- **NVIDIA GPU + drivers** for VRAM numbers. `nvidia-smi` is *optional* — without
  it, VRAM is reported as `unknown` and model operations still function.
- Python **3.10+**.

## Install

Run directly from GitHub with [uv](https://docs.astral.sh/uv/) (no install needed;
the package is not on PyPI):

```bash
uvx --from git+https://github.com/sushiHex/vram-mcp vram-mcp
```

Or install from source for development:

```bash
git clone https://github.com/sushiHex/vram-mcp
cd vram-mcp
pip install -e .
```

## Run

```bash
vram-mcp
```

The server speaks MCP over stdio, so it is normally launched by an MCP client
rather than by hand.

## MCP client config

Add this to your MCP client's `mcpServers` config (e.g. Claude Code / Claude
Desktop):

```json
{
  "mcpServers": {
    "vram": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sushiHex/vram-mcp", "vram-mcp"]
    }
  }
}
```

Or, with a source checkout installed via `pip install -e .`, point `command`
directly at the installed `vram-mcp` script.

## Configuration

- `OLLAMA_BASE_URL` — Ollama endpoint. Defaults to `http://127.0.0.1:11434`.
- `VRAM_MCP_AUDIT` — disappearance detection + the perf-counter-based process
  table (`other_processes` size/name/cmdline on Windows). Defaults on; set to
  `"0"` to disable both and skip the ~1 s Windows perf-counter cost on every
  `vram_status()`/`list_loaded()` call.
- `VRAM_MCP_MEANINGFUL_MB` — minimum dedicated VRAM (MB) for a non-Ollama
  process to be tracked by disappearance detection. Defaults to `512`.
- `VRAM_MCP_EVENT_CAP` — maximum events retained in `events.jsonl` (oldest
  pruned first). Defaults to `5000`.

## Multi-session coordination

Since every session runs its own `vram-mcp` process, coordination happens via:

- **Claims** — a shared, file-based ledger (`~/.cache/vram-mcp/claims.json`) recording who's using a model and why. Call `claim()` when you start relying on a model; `renew()` periodically if still in use. An un-renewed claim simply expires — no cleanup needed if your session ends unexpectedly.
- **Busy detection** — best-effort, via NVML's per-process GPU utilization (not point-in-time; reads a short recent window so brief gaps between tokens don't misread as idle). Requires no changes to how you call Ollama — it's entirely on vram-mcp's side.
- **Protection** — `unload()`/`ensure_free()` refuse to evict a model that's claimed OR busy, by default. Pass `force=True` when you've already decided it's worth it.

Requires the `nvidia-ml-py` dependency (installed automatically). Falls back gracefully — `claims`/`busy` report as empty/`null` — on non-NVIDIA GPUs or if NVML is unavailable.

### Audit trail

Every session shares one append-only, bounded log at
`~/.cache/vram-mcp/events.jsonl` (paired with a `~/.cache/vram-mcp/last_seen.json`
baseline). Two kinds of events land there:

- **Actions** — every `unload`/`ensure_free`/`warm` call, tagged with the `by`
  argument you passed (defaults to `"unknown"` if omitted), whether it
  succeeded/failed/was refused, and why.
- **Disappearances/appearances** — on every `vram_status()`/`list_loaded()`
  call, vram-mcp diffs the current set of "meaningful" VRAM holders (every
  Ollama model, plus any other process using at least `VRAM_MCP_MEANINGFUL_MB`
  of dedicated VRAM) against the previous snapshot and logs what changed, with
  a best-effort **cause**:
  - `self_action` — a recent `unload`/`ensure_free` action from *this* log
    explains the disappearance.
  - `external` — no matching vram-mcp action; likely Ollama idle-expiry,
    memory-pressure eviction, or an unload issued outside vram-mcp.
  - `unattributed` — a non-Ollama process disappeared; vram-mcp can't observe
    why a process exited.

Call `history(model=None, type=None, limit=50, since=None)` to query it — e.g.
"what happened to `llama3`?" or "did anything unexpectedly vanish in the last
hour?". The whole audit path is best-effort: a failure to read or write the log
never breaks a tool call. Disable detection (and its ~1 s Windows perf-counter
cost) with `VRAM_MCP_AUDIT=0`; tune retention with `VRAM_MCP_EVENT_CAP` and the
size threshold with `VRAM_MCP_MEANINGFUL_MB`. Action events
(`unload`/`ensure_free`/`warm`) are always recorded regardless of
`VRAM_MCP_AUDIT`; the variable gates only the passive disappearance/appearance
detection and the perf-counter process table.

`other_processes` (from `vram_status()`) now carries real `size_mb`/`name`/
`cmdline` for every VRAM-holding process, not just Ollama's — on Windows via a
`\GPU Process Memory(*)\Dedicated Usage` perf-counter sample joined with
`Get-CimInstance Win32_Process` (the same source Task Manager uses), since
NVML alone reports `null` sizes under WDDM. That perf-counter sample costs
roughly 1 second per `vram_status()`/`list_loaded()` call; set
`VRAM_MCP_AUDIT=0` to skip it (falls back to NVML's raw process list, sizes
included where NVML can report them, names/cmdlines omitted).

## Development

```bash
pip install -e .
python -m pytest -q
```

The logic modules (`gpu.py`, `ollama.py`, `core.py`, `nvml.py`,
`ollama_correlate.py`, `claims.py`) are free of any `mcp` import and are fully
unit-tested with mocks — no real GPU, Ollama daemon, or `mcp` package required
to run the test suite.

## Roadmap

- Other backends: AMD (ROCm/`rocm-smi`), Intel (`xpu-smi`).
- Other runtimes: vLLM, llama.cpp.

## License

MIT © 2026 sushiHex
