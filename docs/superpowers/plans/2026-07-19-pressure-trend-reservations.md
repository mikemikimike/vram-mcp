# Pressure, Trend, and Reservations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WDDM spill/pressure signal, a throttled free-VRAM trend log, and cooperative GB reservations that gate `warm()`.

**Architecture:** Three independent features layered on the existing pure-module design. `procinfo` widens its single perf-counter sample from one counter to three; `core` gains two pure functions (`pressure`, `reserved_mb`) and one admission predicate (`can_warm`); `audit` gains throttled sampling plus a pure summarizer; `claims` gains reservation records in the existing ledger. `server.py` wires them. No module outside `server.py` imports `mcp`; every reader, clock, and path stays injected.

**Tech Stack:** Python 3.10+, pytest, PowerShell perf counters (Windows), NVML via `nvidia-ml-py`, Ollama HTTP API.

**Spec:** `docs/superpowers/specs/2026-07-19-pressure-trend-reservations-design.md`

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `vram_mcp/procinfo.py` | Modify | Three-counter sample; rows gain `shared_mb`/`non_local_mb` |
| `vram_mcp/core.py` | Modify | `pressure()`, `reserved_mb()`, `can_warm()`; `combined_status` gains `pressure` |
| `vram_mcp/audit.py` | Modify | `maybe_log_sample()`, `summarize_samples()` |
| `vram_mcp/claims.py` | Modify | `reserve()`; `kind` tagging with back-compat |
| `vram_mcp/ollama.py` | Modify | `tags()` for on-disk model sizes |
| `vram_mcp/server.py` | Modify | Wire pressure/sampling/reservations; `trend` + `reserve` tools; `warm` admission |
| `README.md` | Modify | Document new tools + config |
| `tests/test_procinfo.py` | Modify | Three-counter parsing |
| `tests/test_core.py` | Modify | `pressure`, `reserved_mb`, `can_warm` |
| `tests/test_audit.py` | Modify | Sampling throttle + summarizer |
| `tests/test_claims.py` | Modify | Reservations + back-compat |
| `tests/test_ollama.py` | Modify | `tags()` |

---

### Task 1: Three-counter perf sample

**Files:**
- Modify: `vram_mcp/procinfo.py:19-56` (`_WIN_GPU_PS`, `win_gpu_procs`), `:81-112` (`process_table`)
- Test: `tests/test_procinfo.py`

**Context:** Today one `Get-Counter` call samples `\GPU Process Memory(*)\Dedicated Usage` and emits `pid|bytes|name|cmdline`. We widen it to three counters in the *same* call (so the ~1 s cost is unchanged) and emit `pid|dedicated|shared|nonlocal|name|cmdline`. New fields go **before** `cmdline` because a command line may itself contain `|` and the parser uses a bounded `split`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_procinfo.py`:

```python
def test_win_gpu_procs_parses_three_counters(monkeypatch):
    out = (
        "1328|948961280|104857600|0|dwm.exe|C:\\Windows\\dwm.exe\n"
        "28384|18229198848|2097152|1073741824|python.exe|python train.py --a b\n"
    )
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    rows = {r["pid"]: r for r in procinfo.win_gpu_procs()}
    assert rows[1328]["size_mb"] == 905
    assert rows[1328]["shared_mb"] == 100
    assert rows[1328]["non_local_mb"] == 0
    assert rows[28384]["size_mb"] == 17385
    assert rows[28384]["non_local_mb"] == 1024
    assert rows[28384]["name"] == "python.exe"


def test_win_gpu_procs_keeps_pipe_in_cmdline(monkeypatch):
    out = "42|1048576|0|0|sh.exe|sh -c 'a | b | c'\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    (row,) = procinfo.win_gpu_procs()
    assert row["cmdline"] == "sh -c 'a | b | c'"


def test_win_gpu_procs_skips_malformed_lines(monkeypatch):
    out = "not-a-pid|1|2|3|x|y\n7|1048576|0|0|a.exe|a\nshort|line\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    assert [r["pid"] for r in procinfo.win_gpu_procs()] == [7]


def test_process_table_merges_spill_fields():
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": None, "kind": "compute"}],
        win_gpu_reader=lambda: [
            {"pid": 7, "size_mb": 500, "shared_mb": 12, "non_local_mb": 300,
             "name": "a.exe", "cmdline": "a"},
        ],
    )
    assert rows[0]["non_local_mb"] == 300
    assert rows[0]["shared_mb"] == 12
    assert rows[0]["size_mb"] == 500


def test_process_table_defaults_spill_fields_to_none():
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": 100, "kind": "compute"}],
    )
    assert rows[0]["shared_mb"] is None
    assert rows[0]["non_local_mb"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_procinfo.py -q`
Expected: FAIL — `KeyError: 'shared_mb'`.

- [ ] **Step 3: Replace `_WIN_GPU_PS` and `win_gpu_procs`**

In `vram_mcp/procinfo.py`, replace the `_WIN_GPU_PS` constant with:

```python
# ONE Get-Counter call sampling all three counters, so the ~1 s perf-counter
# cost is paid once. Samples are discriminated by their Path (which is
# lowercased by the counter subsystem; -match is case-insensitive anyway).
_WIN_GPU_PS = (
    "$paths=@('\\GPU Process Memory(*)\\Dedicated Usage',"
    "'\\GPU Process Memory(*)\\Shared Usage',"
    "'\\GPU Process Memory(*)\\Non Local Usage');"
    "$d=@{};$s=@{};$n=@{};"
    "(Get-Counter -Counter $paths -EA SilentlyContinue).CounterSamples | "
    "Where-Object { $_.CookedValue -gt 0 -and $_.InstanceName -match 'pid_(\\d+)' } | "
    "ForEach-Object { "
    "$id=[int]($_.InstanceName -replace '.*pid_(\\d+).*','$1'); "
    "$v=[int64]$_.CookedValue; "
    "if($_.Path -match 'non local usage'){ $n[$id]=[int64]$n[$id]+$v } "
    "elseif($_.Path -match 'shared usage'){ $s[$id]=[int64]$s[$id]+$v } "
    "else { $d[$id]=[int64]$d[$id]+$v } };"
    "$ids=@($d.Keys)+@($s.Keys)+@($n.Keys) | Sort-Object -Unique;"
    "foreach($id in $ids){ $p=Get-CimInstance Win32_Process -Filter "
    "\"ProcessId=$id\" -EA SilentlyContinue; "
    "'{0}|{1}|{2}|{3}|{4}|{5}' -f "
    "$id,[int64]$d[$id],[int64]$s[$id],[int64]$n[$id],$p.Name,$p.CommandLine }"
)
```

Replace the body of `win_gpu_procs` (keep its signature and the `sys.platform`
guard) with:

```python
    procs = []
    for line in out_text.splitlines():
        parts = line.strip().split("|", 5)
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        pid, dedicated, shared, non_local, name, cmdline = parts
        procs.append({
            "pid": int(pid),
            "size_mb": bytes_to_mb(dedicated, default=None),
            "shared_mb": bytes_to_mb(shared, default=None),
            "non_local_mb": bytes_to_mb(non_local, default=None),
            "name": name.strip() or None,
            "cmdline": cmdline.strip() or None,
        })
    return procs
```

Update the `win_gpu_procs` docstring to
`"""``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline}]`` for every VRAM holder on Windows.``[]`` off-Windows or on any failure. ~1 s (one three-counter perf sample)."""`

- [ ] **Step 4: Update `process_table` to carry the new fields**

In `process_table`, change the NVML seeding dict to include the new keys:

```python
        table[p["pid"]] = {
            "pid": p["pid"], "size_mb": p.get("size_mb"),
            "shared_mb": None, "non_local_mb": None,
            "name": None, "cmdline": None, "kind": p.get("kind", "compute"),
        }
```

and in the `win_gpu_reader` branch, change the missing-entry default and the
assignment block:

```python
            if entry is None:
                entry = {"pid": w["pid"], "size_mb": None, "shared_mb": None,
                         "non_local_mb": None, "name": None, "cmdline": None,
                         "kind": "compute"}
                table[w["pid"]] = entry
            entry["size_mb"] = w.get("size_mb")
            entry["shared_mb"] = w.get("shared_mb")
            entry["non_local_mb"] = w.get("non_local_mb")
            entry["name"] = w.get("name")
            entry["cmdline"] = w.get("cmdline")
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_procinfo.py -q`
Expected: PASS (all, including the pre-existing tests).

- [ ] **Step 6: Verify live on the real GPU**

Run:
```bash
python -c "from vram_mcp import procinfo, nvml; rows=procinfo.process_table(nvml_processes=nvml.nvml_processes, win_gpu_reader=procinfo.win_gpu_procs); print([(r['name'], r['size_mb'], r['shared_mb'], r['non_local_mb']) for r in rows if r['size_mb']][:5])"
```
Expected: real process names with non-null `size_mb` and integer `shared_mb`.
If every `shared_mb` is `None`, the PowerShell is broken — fix before committing.

- [ ] **Step 7: Commit**

```bash
git add vram_mcp/procinfo.py tests/test_procinfo.py
git commit -m "feat(procinfo): sample Shared + Non Local usage in the same perf-counter call"
```

---

### Task 2: `core.pressure()`

**Files:**
- Modify: `vram_mcp/core.py` (add after `other_processes`, and extend `combined_status`)
- Test: `tests/test_core.py`

**Context:** Two independent slow-modes must be distinguished: driver-forced
spill (`non_local_mb`, the 7× mode) and deliberate Ollama CPU offload
(`offloaded_to_cpu`). Verified live that these do not co-occur, so reporting
only the second is insufficient.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core.py`:

```python
GPUS_OK = [{"index": 0, "total_mb": 24576, "used_mb": 4000, "free_mb": 20576}]
GPUS_TIGHT = [{"index": 0, "total_mb": 24576, "used_mb": 24400, "free_mb": 176}]


def test_pressure_ok():
    p = core.pressure(GPUS_OK, [], [])
    assert p["state"] == "ok"
    assert p["spilling"] is False
    assert p["non_local_mb"] == 0


def test_pressure_tight_when_free_low():
    assert core.pressure(GPUS_TIGHT, [], [])["state"] == "tight"


def test_pressure_degraded_on_cpu_offload():
    loaded = [{"name": "qwen3:32b", "offloaded_to_cpu": True}]
    p = core.pressure(GPUS_OK, loaded, [])
    assert p["state"] == "degraded"
    assert p["offloaded_models"] == ["qwen3:32b"]


def test_pressure_thrashing_beats_degraded():
    loaded = [{"name": "qwen3:32b", "offloaded_to_cpu": True}]
    procs = [{"pid": 1, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, loaded, procs)
    assert p["state"] == "thrashing"
    assert p["spilling"] is True
    assert p["non_local_mb"] == 4096


def test_pressure_ignores_noise_below_threshold():
    procs = [{"pid": 1, "non_local_mb": 10}, {"pid": 2, "non_local_mb": 20}]
    p = core.pressure(GPUS_OK, [], procs)
    assert p["non_local_mb"] == 30
    assert p["spilling"] is False
    assert p["state"] == "ok"


def test_pressure_tolerates_none_values():
    procs = [{"pid": 1, "non_local_mb": None}, {"pid": 2}]
    p = core.pressure([{"index": 0, "free_mb": None}], [{"name": None}], procs)
    assert p["non_local_mb"] == 0
    assert p["free_mb"] is None
    assert p["state"] == "ok"


def test_combined_status_includes_pressure():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [],
    )
    assert status["pressure"]["state"] == "ok"
```

> Note: `FakeOllama` already exists in `tests/test_core.py`. Reuse it; if its
> constructor differs, match the existing call sites in that file rather than
> introducing a new fake.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -q -k pressure`
Expected: FAIL with `AttributeError: module 'vram_mcp.core' has no attribute 'pressure'`.

- [ ] **Step 3: Implement `pressure`**

In `vram_mcp/core.py`, add the constants near `_MB_PER_GB`:

```python
SPILL_THRESHOLD_MB = 256   # below this, non-local usage is normal desktop noise
TIGHT_MB = 1024
```

and add this function after `other_processes`:

```python
def pressure(gpus: list[dict], loaded: list[dict], other_processes: list[dict],
             *, spill_threshold_mb: int = SPILL_THRESHOLD_MB,
             tight_mb: int = TIGHT_MB) -> dict:
    """Classify how close the GPU is to a slow-mode, and which one.

    Two independent degradations exist and must not be conflated:

    * **Driver-forced spill** — WDDM demand-pages VRAM to system RAM under
      pressure (``non_local_mb``). This is the severe multi-x slowdown.
    * **Deliberate CPU offload** — Ollama chose to put layers on the CPU
      backend up front (``offloaded_to_cpu``). Slower, but a considered choice.

    Returns ``{"state", "free_mb", "non_local_mb", "spilling",
    "offloaded_models", "detail"}``. ``state`` is the first match of
    thrashing > degraded > tight > ok.
    """
    free_mb = _gpu.max_free_mb(gpus)
    non_local_mb = sum(
        p.get("non_local_mb") or 0
        for p in other_processes if isinstance(p, dict)
    )
    spilling = non_local_mb >= spill_threshold_mb
    offloaded = [
        m["name"] for m in loaded
        if isinstance(m, dict) and m.get("offloaded_to_cpu") and m.get("name")
    ]

    if spilling:
        state = "thrashing"
        detail = (
            f"{non_local_mb} MB of VRAM has spilled to system RAM; the driver "
            "is paging. Expect severe slowdown — free VRAM or reduce load."
        )
    elif offloaded:
        state = "degraded"
        detail = (
            f"Model(s) partly on CPU: {', '.join(offloaded)}. Slower than "
            "full GPU residency, but a deliberate Ollama placement, not paging."
        )
    elif free_mb is not None and free_mb < tight_mb:
        state = "tight"
        detail = (
            f"Only {free_mb} MB free; the next load will likely spill or fail."
        )
    else:
        state = "ok"
        detail = "No VRAM pressure detected."

    return {
        "state": state,
        "free_mb": free_mb,
        "non_local_mb": non_local_mb,
        "spilling": spilling,
        "offloaded_models": offloaded,
        "detail": detail,
    }
```

- [ ] **Step 4: Wire it into `combined_status`**

In `combined_status`, after the `other_processes` block, replace the trailing
`return result` region so it reads:

```python
    source = procinfo_fn if procinfo_fn is not None else nvml_processes_fn
    procs: list[dict] = []
    if source is not None:
        procs = other_processes(source, resolved_pids)
        result["other_processes"] = procs
    result["pressure"] = pressure(gpus, loaded, procs)
    return result
```

Update the `combined_status` docstring to mention that the result always
carries a `pressure` key, computed from the process table when one is
available (spill detection needs it) and from GPU + model data otherwise.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_core.py -q`
Expected: PASS. If a pre-existing test asserts the exact set of keys in
`combined_status`, update that assertion to include `pressure`.

- [ ] **Step 6: Commit**

```bash
git add vram_mcp/core.py tests/test_core.py
git commit -m "feat(core): pressure() distinguishes driver spill from deliberate CPU offload"
```

---

### Task 3: Throttled trend sampling

**Files:**
- Modify: `vram_mcp/audit.py`
- Test: `tests/test_audit.py`

**Context:** `events.jsonl` is capped at 5000 events. An unthrottled sample on
every status call would evict the action/disappearance events that carry the
real diagnostic value, so sampling is rate-limited across all sessions via a
small shared state file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_audit.py`:

```python
def _sample(free_mb=1000, state="ok"):
    return {"free_mb": free_mb, "used_mb": 23000, "total_mb": 24576,
            "non_local_mb": 0, "state": state, "loaded_count": 1}


def test_first_sample_is_always_written(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now) is True
    rows = audit.read_events(type="sample", path=log)
    assert len(rows) == 1
    assert rows[0]["free_mb"] == 1000


def test_sample_throttled_within_interval(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    t0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    audit.maybe_log_sample(_sample(), state_path=state, log_path=log,
                           now_fn=lambda: t0, interval_seconds=60)
    t1 = t0 + timedelta(seconds=30)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log,
        now_fn=lambda: t1, interval_seconds=60) is False
    assert len(audit.read_events(type="sample", path=log)) == 1


def test_sample_written_after_interval(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    t0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    audit.maybe_log_sample(_sample(), state_path=state, log_path=log,
                           now_fn=lambda: t0, interval_seconds=60)
    t1 = t0 + timedelta(seconds=61)
    assert audit.maybe_log_sample(
        _sample(free_mb=200), state_path=state, log_path=log,
        now_fn=lambda: t1, interval_seconds=60) is True
    assert len(audit.read_events(type="sample", path=log)) == 2


def test_corrupt_state_file_treated_as_never_sampled(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    state.write_text("{not json", encoding="utf-8")
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now) is True


def test_maybe_log_sample_never_raises(tmp_path):
    # A directory where the state file should be makes every write fail.
    state = tmp_path / "s.json"
    state.mkdir()
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=tmp_path / "e.jsonl") is False


def test_summarize_samples_reports_falling_trend():
    rows = [{"free_mb": v, "state": "ok"} for v in (9000, 8000, 3000, 1000)]
    s = audit.summarize_samples(rows)
    assert s["direction"] == "falling"
    assert s["min_free_mb"] == 1000
    assert s["max_free_mb"] == 9000
    assert s["latest_free_mb"] == 1000
    assert s["count"] == 4


def test_summarize_samples_counts_spilling():
    rows = [{"free_mb": 100, "state": "thrashing"},
            {"free_mb": 100, "state": "ok"},
            {"free_mb": 100, "state": "thrashing"}]
    assert audit.summarize_samples(rows)["thrashing_samples"] == 2


def test_summarize_samples_handles_empty_and_none():
    assert audit.summarize_samples([])["count"] == 0
    assert audit.summarize_samples([{"free_mb": None}])["direction"] == "unknown"
```

Ensure `tests/test_audit.py` imports `timedelta` — add it to the existing
`from datetime import ...` line if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audit.py -q -k sample`
Expected: FAIL with `AttributeError: module 'vram_mcp.audit' has no attribute 'maybe_log_sample'`.

- [ ] **Step 3: Implement sampling**

In `vram_mcp/audit.py`, add next to the other path constants:

```python
DEFAULT_SAMPLE_PATH = Path.home() / ".cache" / "vram-mcp" / "last_sample.json"
SAMPLE_INTERVAL_SECONDS = 60.0
```

and add these two functions:

```python
def maybe_log_sample(sample: dict, *, interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
                     state_path: Path = DEFAULT_SAMPLE_PATH,
                     log_path: Path = DEFAULT_EVENTS_PATH,
                     now_fn=_default_now, cap: int = 5000) -> bool:
    """Append one ``type="sample"`` event, at most once per ``interval_seconds``
    across all sessions.

    The throttle matters: ``events.jsonl`` is capped, and an unthrottled sample
    on every status call would evict the action/disappearance events that carry
    the real diagnostic value. The last-sample timestamp lives in its own small
    state file under the shared lock, so concurrent sessions agree on the rate.

    Returns True if a sample was written. Best-effort — never raises.
    """
    try:
        now = now_fn()
        with locked(state_path):
            state = load_json(state_path, lambda: {},
                              lambda d: isinstance(d, dict))
            last = state.get("ts")
            if last:
                try:
                    if (now - parse_iso(last)).total_seconds() < interval_seconds:
                        return False
                except (ValueError, TypeError):
                    pass  # unparsable timestamp -> treat as never sampled
            save_json_atomic(state_path, {"ts": iso(now)})
    except Exception:
        return False
    # Written outside the state lock, under the events lock, so every writer of
    # events.jsonl is serialized by the same lock and none are nested.
    event = {"ts": iso(now), "type": "sample", **sample}
    try:
        with locked(log_path):
            append_jsonl_capped(log_path, event, cap)
    except Exception:
        return False
    return True


def summarize_samples(rows: list[dict]) -> dict:
    """Reduce ``type="sample"`` events to a trend. Pure — no IO.

    ``direction`` compares the mean of the first third against the last third,
    which is robust to a single spike in a way that first-vs-last is not.
    Rows are expected oldest-first.
    """
    values = [r.get("free_mb") for r in rows
              if isinstance(r, dict) and isinstance(r.get("free_mb"), int)]
    thrashing = sum(1 for r in rows
                    if isinstance(r, dict) and r.get("state") == "thrashing")
    if not values:
        return {"count": len(rows), "direction": "unknown",
                "min_free_mb": None, "max_free_mb": None,
                "latest_free_mb": None, "thrashing_samples": thrashing}

    third = max(1, len(values) // 3)
    head = sum(values[:third]) / third
    tail = sum(values[-third:]) / third
    delta = tail - head
    # 10% of the starting level, floored at 128 MB, so ordinary jitter on a
    # quiet GPU doesn't read as a trend.
    threshold = max(128.0, abs(head) * 0.10)
    if delta <= -threshold:
        direction = "falling"
    elif delta >= threshold:
        direction = "rising"
    else:
        direction = "flat"

    return {
        "count": len(rows),
        "direction": direction,
        "min_free_mb": min(values),
        "max_free_mb": max(values),
        "latest_free_mb": values[-1],
        "thrashing_samples": thrashing,
    }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_audit.py -q`
Expected: PASS (all, including the pre-existing audit tests).

- [ ] **Step 5: Commit**

```bash
git add vram_mcp/audit.py tests/test_audit.py
git commit -m "feat(audit): throttled free-VRAM sampling + pure trend summarizer"
```

---

### Task 4: Reservations in the claims ledger

**Files:**
- Modify: `vram_mcp/claims.py`, `vram_mcp/core.py`
- Test: `tests/test_claims.py`, `tests/test_core.py`

**Context:** A reservation claims **GB of VRAM**, not a named model. It reuses
the existing ledger for identical TTL/crash-safety semantics. Existing records
have no `kind` field and must keep working — absent `kind` means `"model"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_claims.py`:

```python
def test_reserve_round_trip(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    r = claims.reserve(8.0, "trainer", "dpo run", 3600,
                       pid=1234, path=path, now_fn=lambda: now)
    assert r["claim_id"]
    (rec,) = claims.list_claims(path=path, now_fn=lambda: now)
    assert rec["kind"] == "reservation"
    assert rec["gb"] == 8.0
    assert rec["pid"] == 1234
    assert rec["model"] is None


def test_reservation_expires_by_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.reserve(8.0, "trainer", "dpo", 60, path=path, now_fn=lambda: now)
    later = now + timedelta(seconds=61)
    assert claims.list_claims(path=path, now_fn=lambda: later) == []


def test_claim_records_are_tagged_model(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.claim("llama3", "me", "chat", 3600, path=path, now_fn=lambda: now)
    (rec,) = claims.list_claims(path=path, now_fn=lambda: now)
    assert rec["kind"] == "model"


def test_list_claims_by_model_excludes_reservations(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.claim("llama3", "me", "chat", 3600, path=path, now_fn=lambda: now)
    claims.reserve(8.0, "trainer", "dpo", 3600, path=path, now_fn=lambda: now)
    got = claims.list_claims("llama3", path=path, now_fn=lambda: now)
    assert len(got) == 1
    assert got[0]["kind"] == "model"


def test_legacy_record_without_kind_still_lists(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    expires = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps({"claims": [
        {"claim_id": "old", "model": "llama3", "owner": "me",
         "purpose": "chat", "ttl_seconds": 3600, "expires_at": expires},
    ]}), encoding="utf-8")
    got = claims.list_claims("llama3", path=path, now_fn=lambda: now)
    assert len(got) == 1
```

Add to `tests/test_core.py`:

```python
def test_reserved_mb_sums_active_reservations():
    recs = [{"kind": "reservation", "gb": 8.0},
            {"kind": "reservation", "gb": 2.5},
            {"kind": "model", "model": "llama3"}]
    assert core.reserved_mb(recs) == int(round(10.5 * 1024))


def test_reserved_mb_skips_malformed():
    recs = [{"kind": "reservation", "gb": "eight"},
            {"kind": "reservation"},
            "not-a-dict",
            {"kind": "reservation", "gb": 1.0}]
    assert core.reserved_mb(recs) == 1024


def test_reserved_mb_empty():
    assert core.reserved_mb([]) == 0
```

Ensure `tests/test_claims.py` imports `json` and `timedelta`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_claims.py tests/test_core.py -q -k "reserv"`
Expected: FAIL with `AttributeError: module 'vram_mcp.claims' has no attribute 'reserve'`.

- [ ] **Step 3: Implement `reserve` and tag `claim`**

In `vram_mcp/claims.py`, add `"kind": "model",` to the record dict built inside
`claim()` (right after `"claim_id"`), then add this function after `claim()`:

```python
def reserve(
    gb: float, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    pid: Optional[int] = None, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Reserve ``gb`` GB of VRAM for ``owner`` — a claim on capacity rather
    than on a named model.

    Reservations share the ledger (and therefore the TTL and crash-safety
    semantics) with model claims, tagged ``kind="reservation"`` and carrying
    ``model=None`` so they never shadow or protect a model. ``pid`` is
    advisory: it records which process the reservation is for.

    Returns ``{"claim_id", "expires_at"}``.
    """
    path = path or _DEFAULT_PATH
    now = now_fn()
    expires_at = now + timedelta(seconds=ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "reservation",
        "model": None, "gb": float(gb), "pid": pid,
        "owner": owner, "purpose": purpose,
        "claimed_at": _iso(now), "renewed_at": _iso(now),
        "ttl_seconds": ttl_seconds, "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now)
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}
```

Then make `list_claims`'s model filter reservation-safe — replace

```python
        active = [r for r in active if r["model"] == model]
```

with

```python
        active = [r for r in active if r.get("model") == model]
```

and extend that method's docstring with: *"Reservations (``kind="reservation"``,
``model=None``) are returned by an unfiltered call and excluded by a
``model=`` filter. Records written by older versions carry no ``kind`` and are
treated as model claims."*

- [ ] **Step 4: Implement `core.reserved_mb`**

In `vram_mcp/core.py`, add after `is_protected`:

```python
def reserved_mb(all_claims: list[dict]) -> int:
    """Total VRAM (MB) spoken for by active reservations.

    ``all_claims`` is the ledger's already-expiry-filtered list, so every
    reservation here is live. Malformed records are skipped rather than
    raising — an unusable record must not break a status call.
    """
    total = 0.0
    for record in all_claims:
        if not isinstance(record, dict) or record.get("kind") != "reservation":
            continue
        try:
            total += float(record["gb"])
        except (KeyError, TypeError, ValueError):
            continue
    return int(round(total * _MB_PER_GB))
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_claims.py tests/test_core.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vram_mcp/claims.py vram_mcp/core.py tests/test_claims.py tests/test_core.py
git commit -m "feat(claims): GB reservations in the shared ledger + core.reserved_mb"
```

---

### Task 5: `ollama.tags()` and the `can_warm` admission predicate

**Files:**
- Modify: `vram_mcp/ollama.py`, `vram_mcp/core.py`
- Test: `tests/test_ollama.py`, `tests/test_core.py`

**Context:** To decide whether a warm fits, we need the model's size before it
loads. `/api/tags` reports on-disk size, which approximates VRAM need. It is an
approximation, so it only ever drives an **overridable** refusal.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ollama.py` (match the existing transport-faking style in
that file — the tests below assume `_get_json`; if the client's read helper has
a different name, fake that one instead):

```python
def test_tags_returns_model_sizes(monkeypatch):
    client = OllamaClient(base_url="http://x")
    payload = {"models": [
        {"name": "llama3:8b", "size": 4 * 1024 * 1024 * 1024},
        {"name": "qwen3:32b", "size": 20 * 1024 * 1024 * 1024},
    ]}
    monkeypatch.setattr(client, "_get_json", lambda path: payload)
    assert client.tags()["llama3:8b"] == 4096
    assert client.tags()["qwen3:32b"] == 20480


def test_tags_empty_on_failure(monkeypatch):
    client = OllamaClient(base_url="http://x")
    monkeypatch.setattr(client, "_get_json", lambda path: None)
    assert client.tags() == {}


def test_tags_skips_nameless_rows(monkeypatch):
    client = OllamaClient(base_url="http://x")
    monkeypatch.setattr(
        client, "_get_json",
        lambda path: {"models": [{"size": 1024}, {"name": "a", "size": 1048576}]})
    assert client.tags() == {"a": 1}
```

Add to `tests/test_core.py`:

```python
def test_can_warm_allows_when_it_fits():
    ok, detail = core.can_warm("llama3", free_mb=20000, reserved_mb=8192,
                               model_size_mb=4096)
    assert ok is True
    assert detail["headroom_mb"] == 20000 - 8192


def test_can_warm_refuses_when_reservations_consume_headroom():
    ok, detail = core.can_warm("qwen3:32b", free_mb=10000, reserved_mb=8192,
                               model_size_mb=20480)
    assert ok is False
    assert detail["reason"] == "insufficient_headroom"
    assert detail["model_size_mb"] == 20480


def test_can_warm_refuses_when_headroom_exhausted_and_size_unknown():
    ok, detail = core.can_warm("mystery", free_mb=4000, reserved_mb=8192,
                               model_size_mb=None)
    assert ok is False
    assert detail["reason"] == "no_headroom"


def test_can_warm_allows_unknown_size_with_headroom():
    ok, _ = core.can_warm("mystery", free_mb=20000, reserved_mb=1024,
                          model_size_mb=None)
    assert ok is True


def test_can_warm_allows_when_free_unknown():
    # VRAM unreadable -> we cannot prove it will not fit; never block on a guess.
    ok, detail = core.can_warm("m", free_mb=None, reserved_mb=8192,
                               model_size_mb=4096)
    assert ok is True
    assert detail["reason"] == "free_unknown"


def test_can_warm_allows_when_nothing_reserved():
    ok, _ = core.can_warm("m", free_mb=100, reserved_mb=0, model_size_mb=99999)
    assert ok is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ollama.py tests/test_core.py -q -k "tags or can_warm"`
Expected: FAIL — `tags` / `can_warm` do not exist.

- [ ] **Step 3: Implement `OllamaClient.tags()`**

Read `vram_mcp/ollama.py:36-50` (`ps`) first and mirror its transport and
failure handling exactly — `tags()` must degrade to `{}` on any failure the
same way `ps()` degrades. Add:

```python
    def tags(self) -> dict:
        """``{model_name: size_mb}`` from ``/api/tags`` (on-disk sizes).

        Disk size approximates VRAM need — close enough to decide whether a
        warm plausibly fits, never precise enough to be authoritative. ``{}``
        on any failure, so callers must treat a missing entry as "unknown"
        rather than "zero".
        """
        payload = self._get_json("/api/tags")
        if not isinstance(payload, dict):
            return {}
        sizes = {}
        for row in payload.get("models") or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if not name:
                continue
            size = _shared_bytes_to_mb(row.get("size"), default=None)
            if size is not None:
                sizes[name] = size
        return sizes
```

If `ollama.py` does not already have a `_get_json` helper, extract one from
`ps()` so both methods share a single transport + failure path rather than
duplicating it, and import `bytes_to_mb as _shared_bytes_to_mb` from `._util`.

- [ ] **Step 4: Implement `core.can_warm`**

In `vram_mcp/core.py`, add after `reserved_mb`:

```python
def can_warm(model: str, *, free_mb, reserved_mb: int, model_size_mb) -> tuple[bool, dict]:
    """May ``model`` be warmed without eating VRAM another session reserved?

    Cooperative, not enforced: vram-mcp cannot intercept an Ollama auto-load
    triggered by a direct ``/api/generate`` call, so this gates only vram-mcp's
    own ``warm()``. Every refusal is overridable with ``force=True``.

    Refuses when reservations leave no headroom at all, or when the model's
    approximate size exceeds the headroom. Never refuses on a guess: unknown
    free VRAM always allows.

    Returns ``(allowed, detail)`` where detail carries ``reason``,
    ``headroom_mb``, ``reserved_mb`` and ``model_size_mb``.
    """
    base = {"reserved_mb": reserved_mb, "model_size_mb": model_size_mb,
            "free_mb": free_mb}
    if free_mb is None:
        return True, {**base, "headroom_mb": None, "reason": "free_unknown"}

    headroom = free_mb - reserved_mb
    detail = {**base, "headroom_mb": headroom}
    if reserved_mb <= 0:
        return True, {**detail, "reason": "no_reservations"}
    if headroom <= 0:
        return False, {**detail, "reason": "no_headroom"}
    if model_size_mb is not None and model_size_mb > headroom:
        return False, {**detail, "reason": "insufficient_headroom"}
    return True, {**detail, "reason": "fits"}
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_ollama.py tests/test_core.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vram_mcp/ollama.py vram_mcp/core.py tests/test_ollama.py tests/test_core.py
git commit -m "feat: ollama.tags() + core.can_warm admission predicate"
```

---

### Task 6: Server wiring, new tools, and docs

**Files:**
- Modify: `vram_mcp/server.py`, `README.md`

**Context:** Final task — expose everything. `server.py` is the only module
that may import `mcp`. Every tool stays `async def` delegating to a sync
`_x_impl` via `_in_thread`, because the installed FastMCP calls sync tools
directly on the event loop.

- [ ] **Step 1: Add config and the sampling hook**

In `vram_mcp/server.py`, add to the config block near `_EVENT_CAP`:

```python
_SAMPLE_SECONDS = float(os.environ.get("VRAM_MCP_SAMPLE_SECONDS", "60"))
_SPILL_MB = int(os.environ.get("VRAM_MCP_SPILL_MB", "256"))
```

Extend `_run_detection` so the same guarded, best-effort block also samples:

```python
def _run_detection(status: dict) -> None:
    """Diff the meaningful-holder set from a status snapshot, log changes, and
    record a throttled free-VRAM sample. Best-effort; disabled by
    VRAM_MCP_AUDIT=0 — never raises (the audit may never break a tool call)."""
    if not _AUDIT_ON:
        return
    try:
        holders = _audit.meaningful_holders(
            status.get("loaded", []), status.get("other_processes", []), _MEANINGFUL_MB)
        _audit.detect_and_log(holders, cap=_EVENT_CAP)
    except Exception:
        pass
    try:
        p = status.get("pressure") or {}
        gpus = status.get("gpus") or []
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
```

- [ ] **Step 2: Surface pressure in the status summary**

In `_vram_status_impl`, replace the summary assignment with:

```python
    p = status.get("pressure") or {}
    state = p.get("state", "unknown")
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}, pressure: {state}."
        + (f" {p['detail']}" if state not in ("ok", "unknown") else "")
    )
```

Update the `vram_status` docstring to mention the `pressure` key (state,
spill detection, and CPU-offload list).

- [ ] **Step 3: Add the `trend` tool**

Add after the `history` tool:

```python
def _trend_impl(hours: float) -> dict:
    since = _audit.iso(_audit._default_now() - timedelta(hours=hours))
    rows = _audit.read_events(type="sample", limit=10_000, since=since)
    rows.reverse()  # read_events is newest-first; the summarizer wants oldest-first
    summary = _audit.summarize_samples(rows)
    if summary["count"] == 0:
        text = (
            f"No VRAM samples in the last {hours}h. Samples are recorded on "
            "status calls at most once per minute, and are disabled by "
            "VRAM_MCP_AUDIT=0."
        )
    else:
        text = (
            f"{summary['count']} sample(s) over {hours}h: free VRAM is "
            f"{summary['direction']} (min {summary['min_free_mb']} MB, "
            f"max {summary['max_free_mb']} MB, now "
            f"{summary['latest_free_mb']} MB); "
            f"{summary['thrashing_samples']} sample(s) showed VRAM spilling "
            "to system RAM."
        )
    return {**summary, "hours": hours, "samples": rows, "summary": text}


@mcp.tool()
async def trend(hours: float = 1.0) -> dict:
    """Free-VRAM trend over the last ``hours``, from the sampled audit log.

    Answers "was this a gradual erosion or a sudden spike?" — the question a
    point-in-time ``vram_status()`` cannot. Returns direction, min/max/latest
    free MB, how many samples showed driver spill, and the raw samples.
    """
    return await _in_thread(_trend_impl, hours)
```

Add `from datetime import timedelta` to the imports at the top of `server.py`.

- [ ] **Step 4: Add the `reserve` tool**

Add to the claim-tools section, after `_claim_impl`/`claim`:

```python
def _reserve_impl(gb: float, owner: str, purpose: str, ttl_seconds: int,
                  pid: Optional[int]) -> dict:
    outcome = _ledger_call("Reserve", _claims.reserve, gb, owner, purpose,
                           ttl_seconds, pid)
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

    This is COOPERATIVE: it gates vram-mcp's own ``warm()``, but vram-mcp
    cannot intercept an Ollama auto-load triggered by a direct /api/generate
    call from another process.
    """
    return await _in_thread(_reserve_impl, gb, owner, purpose, ttl_seconds, pid)
```

`_claims.reserve` takes `pid` as keyword-only, so `_ledger_call`'s positional
passing will fail. Fix by making the call site explicit — replace the
`_ledger_call` line above with:

```python
    outcome = _ledger_call(
        "Reserve",
        lambda: _claims.reserve(gb, owner, purpose, ttl_seconds, pid=pid),
    )
```

(`_ledger_call(verb, fn, *args)` calls `fn(*args)`; with no extra args a
zero-arg lambda is correct.)

- [ ] **Step 5: Gate `warm()` on reservations**

Replace `_warm_impl` and the `warm` tool with:

```python
def _warm_impl(model: str, keep_alive: str, by: str, force: bool) -> dict:
    if not force:
        status = core.combined_status(gpu_status, _ollama)
        active = _claims.list_claims()
        reserved = core.reserved_mb(active)
        allowed, detail = core.can_warm(
            model, free_mb=status["free_mb"], reserved_mb=reserved,
            model_size_mb=_ollama.tags().get(model),
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
```

- [ ] **Step 6: Report reserved VRAM from `ensure_free`**

In `_ensure_free_impl`, after the `core.ensure_free(...)` call add:

```python
    reserved = core.reserved_mb(_claims.list_claims())
    result["reserved_mb"] = reserved
```

and append to the summary construction, right before `result["summary"] = ...`:

```python
    reserved_note = (
        f" Note: {reserved} MB of the free VRAM is reserved by other sessions."
        if reserved else ""
    )
```

then change the assignment to `result["summary"] = base + declined_note + reserved_note`.

- [ ] **Step 7: Run the full suite and a syntax check**

Run: `python -m pytest -q && python -m pyflakes vram_mcp/`
Expected: all tests PASS, pyflakes silent.

- [ ] **Step 8: Smoke-test the server imports and tools register**

Run:
```bash
python -c "from vram_mcp import server; print(sorted(t.name for t in __import__('anyio').run(server.mcp.list_tools)))"
```
Expected: a list including `reserve`, `trend`, `warm`, `history`, `vram_status`.

- [ ] **Step 9: Update the README**

In `README.md`:
1. Add to the Tools table:
   - `` `trend(hours=1.0)` `` — Free-VRAM trend from the sampled audit log: direction, min/max/latest, and how many samples showed driver spill.
   - `` `reserve(gb, owner, purpose, ttl_seconds=3600, pid=None)` `` — Reserve GB of VRAM for non-Ollama work so other sessions see it's spoken for.
2. Change the `warm` row to note `force` and the reservation refusal.
3. Add to the feature bullets: **Pressure detection** — distinguishes driver-forced VRAM spill to system RAM (the severe slow-mode) from Ollama's deliberate CPU offload.
4. Add to Configuration: `VRAM_MCP_SAMPLE_SECONDS` (default `60`) and `VRAM_MCP_SPILL_MB` (default `256`).
5. Add a short **Reservations** subsection under Multi-session coordination stating plainly that reservations are cooperative — they gate vram-mcp's `warm()` and inform other sessions, but vram-mcp cannot intercept an Ollama auto-load from a direct `/api/generate` call.

- [ ] **Step 10: Commit**

```bash
git add vram_mcp/server.py README.md
git commit -m "feat(server): pressure in status, trend + reserve tools, warm admission control"
```

---

## Self-Review Notes

- Spec coverage: A→Tasks 1-2, B→Tasks 3+6, C→Tasks 4-6. Config vars in Task 6.
- `procinfo` row shape changed; `audit.meaningful_holders` reads only `size_mb`/`name`/`cmdline`/`pid`, so it is unaffected.
- `combined_status` gains a key — Task 2 Step 5 explicitly checks for tests asserting an exact key set.
- `_ledger_call` signature mismatch for keyword-only `pid` is called out and fixed inline in Task 6 Step 4.
