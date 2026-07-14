# vram-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents inspect and
free **NVIDIA GPU VRAM** by managing [Ollama](https://ollama.com) models. Handy
when you juggle several local models across projects on a single GPU and an
agent needs to make room before loading the next one.

- **NVIDIA + Ollama only** for v1.
- Degrades gracefully when `nvidia-smi` is absent: VRAM readings become
  `unknown`, but model list / unload / warm still work.

## Tools

| Tool | Behavior |
| --- | --- |
| `vram_status()` | Per-GPU VRAM (total/used/free) + loaded Ollama models (with claims, busy signal, CPU-offload) + every other VRAM-holding process + best free MB. |
| `list_loaded()` | The models currently resident in VRAM (name, VRAM MB, expiry, claims, busy). |
| `unload(model, force=False)` | Evict one model from VRAM now (`keep_alive=0`). Refuses if claimed/busy unless `force=True`. |
| `ensure_free(gb, force=False)` | Unload models largest-first until at least `gb` GB is free, skipping claimed/busy models unless `force=True`. |
| `warm(model, keep_alive="5m")` | Load/pin a model into VRAM for a duration. |
| `advise()` | Heuristic suggestions (e.g. `OLLAMA_MAX_LOADED_MODELS=1`, finite `OLLAMA_KEEP_ALIVE`). |
| `claim(model, owner, purpose, ttl_seconds=3600)` | Declare you're using a model, so others see who/why before evicting it. |
| `renew(claim_id, ttl_seconds=None)` | Extend a claim before it expires. |
| `release(claim_id)` | Release a claim early. |
| `list_claims(model=None)` | See active claims (all models, or one). |

## Requirements

- **Ollama** running locally (or reachable via `OLLAMA_BASE_URL`).
- **NVIDIA GPU + drivers** for VRAM numbers. `nvidia-smi` is *optional* — without
  it, VRAM is reported as `unknown` and model operations still function.
- Python **3.10+**.

## Install

Run directly with [uv](https://docs.astral.sh/uv/) (no install needed):

```bash
uvx vram-mcp
```

Or install from source for development:

```bash
git clone https://github.com/kfaim/vram-mcp
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
      "args": ["vram-mcp"]
    }
  }
}
```

## Configuration

- `OLLAMA_BASE_URL` — Ollama endpoint. Defaults to `http://127.0.0.1:11434`.

## Multi-session coordination

Since every session runs its own `vram-mcp` process, coordination happens via:

- **Claims** — a shared, file-based ledger (`~/.cache/vram-mcp/claims.json`) recording who's using a model and why. Call `claim()` when you start relying on a model; `renew()` periodically if still in use. An un-renewed claim simply expires — no cleanup needed if your session ends unexpectedly.
- **Busy detection** — best-effort, via NVML's per-process GPU utilization (not point-in-time; reads a short recent window so brief gaps between tokens don't misread as idle). Requires no changes to how you call Ollama — it's entirely on vram-mcp's side.
- **Protection** — `unload()`/`ensure_free()` refuse to evict a model that's claimed OR busy, by default. Pass `force=True` when you've already decided it's worth it.

Requires the `nvidia-ml-py` dependency (installed automatically). Falls back gracefully — `claims`/`busy` report as empty/`null` — on non-NVIDIA GPUs or if NVML is unavailable.

## Development

```bash
pip install -e .
python -m pytest -q
```

The logic modules (`gpu.py`, `ollama.py`, `core.py`) are free of any `mcp`
import and are fully unit-tested with mocks — no real GPU, Ollama daemon, or
`mcp` package required to run the test suite.

## Roadmap

- Other backends: AMD (ROCm/`rocm-smi`), Intel (`xpu-smi`).
- Other runtimes: vLLM, llama.cpp.

## License

MIT © 2026 kfaim
