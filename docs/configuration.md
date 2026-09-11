# Configuration

[Back to the README](../README.md)

## Installing from source

Use Python 3.10+:

```sh
git clone https://github.com/sushiHex/vram-mcp.git
cd vram-mcp
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then install:

```sh
python -m pip install -e .
```

In your client's configuration, replace `uvx` and its arguments with the
**absolute path** to `.venv/bin/vram-mcp` on Linux or
`.venv/Scripts/vram-mcp.exe` on Windows, and an empty `args` list. For example,
Codex's `~/.codex/config.toml` entry becomes:

```toml
[mcp_servers.vram]
command = 'C:\path\to\vram-mcp\.venv\Scripts\vram-mcp.exe'
args = []
```

Replace the example path with your checkout's executable. TOML single-quoted
strings preserve Windows backslashes literally. Reconnect clients after
updating the installation so each starts a fresh server process.

## Environment variables

Set these in the **MCP server's environment** through your client configuration.
Restart/reconnect the server after changing them. Use the same settings across
clients sharing one GPU and ledger.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint. |
| `OLLAMA_MODELS` | `~/.ollama/models` | Local model directory used to correlate manifests with runner processes. Match the directory used by Ollama. |
| `VRAM_MCP_AUDIT` | `1` | Enable passive change detection, enriched process readings, and trend sampling. Set exactly `0` to disable. |
| `VRAM_MCP_MEANINGFUL_MB` | `512` | Minimum dedicated VRAM in MB for tracking a non-Ollama process in appearance/disappearance events. |
| `VRAM_MCP_EVENT_CAP` | `5000` | Maximum retained events, including samples; oldest entries are pruned. |
| `VRAM_MCP_SAMPLE_SECONDS` | `60` | Minimum interval between trend samples, shared across sessions. |
| `VRAM_MCP_SPILL_MB` | `256` | Floor of unexplained non-local memory in MB before spill can be reported. It must also exceed free VRAM when that reading is available. |

For clients using `mcpServers` JSON, add an `env` object to the server entry:

```json
{
  "mcpServers": {
    "vram": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sushiHex/vram-mcp", "vram-mcp"],
      "env": {
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "VRAM_MCP_SAMPLE_SECONDS": "60"
      }
    }
  }
}
```

`advise()` may suggest `OLLAMA_MAX_LOADED_MODELS` or `OLLAMA_KEEP_ALIVE`.
Those belong to the **Ollama server's environment**; setting them only on the
MCP process does not reconfigure an already-running Ollama instance.

### Local and remote Ollama

GPU readings, process inspection, model manifests, and claim files are local
to the MCP server process. `OLLAMA_BASE_URL` can point elsewhere, but it does
not move those collectors. For coherent memory decisions, run the MCP server
alongside Ollama with access to its model directory and runner processes.
A container or different OS user may expose different processes and files.

### Audit cost and coverage

On Windows/WDDM, NVML often cannot report per-process memory sizes. Auditing
adds a performance-counter sample and process lookup to `vram_status()` and
`list_loaded()`, costing roughly one second per call on the measured setup.
The resulting process table supplies sizes and, where available, names and
command lines.

`VRAM_MCP_AUDIT=0` skips that enrichment and disables appearance/disappearance
detection, driver-spill detection, and new trend samples. Status falls back to
NVML process data, with sizes where available. Claims, busy detection, model
operations, and action logging remain enabled. Old audit events remain queryable.

Without the Windows non-local-memory counters, pressure may still report
`ok`, `tight`, or `degraded`, but cannot diagnose `thrashing`. Missing process
information is not evidence that nothing else is using the GPU.

Keep the sample interval high enough that samples do not crowd action events
out of the bounded log. Raising the spill floor can suppress noisy alerts;
it does not create additional GPU capacity.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Client cannot start the server | Confirm Git and uv are installed and visible to the client. Use an absolute `uvx` or installed executable path if its PATH differs from your terminal's. Allow time for the first dependency download. |
| Starting `vram-mcp` appears to hang | It is waiting for MCP input over stdio. Connect through an MCP client. |
| `free_mb` is `null` or `gpus` is empty | Run `nvidia-smi` in the server's environment and check driver availability. Model operations can work without memory telemetry. |
| No loaded models appear | Check `ollama ps` and the configured endpoint. An unreachable Ollama endpoint also produces an empty model list. |
| `busy` is `null` | NVML activity or model-to-process correlation is unavailable. Check access to runner processes and the model directory; keep explicit claims for work in use. |
| `warm` returns `ok=false` | Read `summary` and `reason`: a reservation refusal differs from an Ollama failure. Check that the model is installed and Ollama is reachable; recheck status after a timeout before retrying. |
| `trend()` has no samples | Call `vram_status()` with working GPU telemetry and auditing enabled. `list_loaded()` never samples GPU memory; historical data is not collected in the background. |
| A model is protected after inference ends | Check active claims. Busy detection uses a recent activity window and can lag by a few seconds; recheck before considering an override. |

For registration details, see the [Claude Code MCP guide](https://code.claude.com/docs/en/mcp).
The [uv tools guide](https://docs.astral.sh/uv/guides/tools/) covers Git sources
and isolated tool environments.
