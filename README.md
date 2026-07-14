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
| `vram_status()` | Per-GPU VRAM (total/used/free) + loaded Ollama models + best free MB. |
| `list_loaded()` | The models currently resident in VRAM (name, VRAM MB, expiry). |
| `unload(model)` | Evict one model from VRAM now (`keep_alive=0`). |
| `ensure_free(gb)` | Unload models largest-first until at least `gb` GB is free. |
| `warm(model, keep_alive="5m")` | Load/pin a model into VRAM for a duration. |
| `advise()` | Heuristic suggestions (e.g. `OLLAMA_MAX_LOADED_MODELS=1`, finite `OLLAMA_KEEP_ALIVE`). |

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
