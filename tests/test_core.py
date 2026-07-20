"""Tests for vram_mcp.core — pure orchestration with injected fakes."""

from vram_mcp import core


_MB = 1024 * 1024


def gb_bytes(gb):
    return int(gb * 1024 * _MB)


class FakeOllama:
    """Fake OllamaClient: canned ps() list + records unload() calls."""

    def __init__(self, models=None):
        self._models = models or []
        self.unloaded = []

    def ps(self):
        return list(self._models)

    def unload(self, model):
        self.unloaded.append(model)
        return True


def gpu_fn_const(free_mb):
    """gpu_status_fn returning a single GPU with fixed free VRAM."""
    def _fn():
        if free_mb is None:
            return []
        return [{"index": 0, "name": "GPU", "total_mb": 24000,
                 "used_mb": 24000 - free_mb, "free_mb": free_mb}]
    return _fn


def gpu_fn_sequence(free_values):
    """gpu_status_fn yielding a new free value on each successive call."""
    calls = {"i": 0}

    def _fn():
        i = min(calls["i"], len(free_values) - 1)
        calls["i"] += 1
        val = free_values[i]
        if val is None:
            return []
        return [{"index": 0, "name": "GPU", "total_mb": 24000,
                 "used_mb": 24000 - val, "free_mb": val}]
    return _fn


# ---- combined_status --------------------------------------------------------

def test_combined_status():
    models = [
        {"name": "big", "size": gb_bytes(8), "size_vram": gb_bytes(8),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    status = core.combined_status(gpu_fn_const(6000), FakeOllama(models))
    assert status["free_mb"] == 6000
    assert len(status["gpus"]) == 1
    assert status["loaded"] == [
        {"name": "big", "size_vram_mb": 8192, "total_size_mb": 8192,
         "offloaded_to_cpu": False, "expires_at": "2026-07-11T10:00:00Z"},
    ]


def test_combined_status_no_gpu():
    status = core.combined_status(gpu_fn_const(None), FakeOllama([]))
    assert status["gpus"] == []
    assert status["free_mb"] is None
    assert status["loaded"] == []


# ---- offload detection -------------------------------------------------------

def test_loaded_models_detects_cpu_offload():
    models = [
        {"name": "partial", "size": gb_bytes(10), "size_vram": gb_bytes(6),
         "expires_at": None},
    ]
    status = core.combined_status(gpu_fn_const(4000), FakeOllama(models))
    entry = status["loaded"][0]
    assert entry["total_size_mb"] == 10240
    assert entry["size_vram_mb"] == 6144
    assert entry["offloaded_to_cpu"] is True


# ---- Snapshot ------------------------------------------------------------------

def _snap(all_claims=None, pid_map=None, busy_map=None):
    return core.Snapshot(all_claims or [], pid_map or {}, busy_map or {})


def test_snapshot_capture_runs_each_collector_once():
    calls = {"claims": 0, "pids": 0, "busy": 0}

    def all_claims_fn():
        calls["claims"] += 1
        return [{"model": "m1", "owner": "x"}]

    def pid_map_fn():
        calls["pids"] += 1
        return {"m1": 123, "m2": 456}

    def busy_map_fn(pids):
        calls["busy"] += 1
        assert pids == [123, 456]        # exactly the pids the map surfaced
        return {123: True, 456: False}

    snap = core.Snapshot.capture(all_claims_fn, pid_map_fn, busy_map_fn)
    assert calls == {"claims": 1, "pids": 1, "busy": 1}
    assert snap.busy_for("m1") is True
    assert snap.busy_for("m2") is False


def test_snapshot_capture_skips_busy_fetch_when_no_pids():
    def busy_map_fn(pids):
        raise AssertionError("must not be called with no pids")

    snap = core.Snapshot.capture(lambda: [], lambda: {}, busy_map_fn)
    assert snap.busy_for("anything") is None


def test_snapshot_none_name_gets_no_claims_and_no_busy():
    # A nameless ps() row must NEVER be attributed everyone's claims or a pid.
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    assert snap.claims_for(None) == []
    assert snap.pid_for(None) is None
    assert snap.busy_for(None) is None


# ---- attach_coordination ---------------------------------------------------------

def test_attach_coordination_adds_claims_and_busy():
    loaded = [{"name": "m1"}, {"name": "m2"}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    result, resolved = core.attach_coordination(loaded, snap)
    assert result[0]["claims"] == [{"model": "m1", "owner": "x"}]
    assert result[0]["busy"] is True
    assert result[1]["claims"] == []
    assert result[1]["busy"] is None   # no pid -> undetermined, never guessed
    assert resolved == {123}


def test_attach_coordination_nameless_row_stays_unattributed():
    loaded = [{"name": None}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    result, resolved = core.attach_coordination(loaded, snap)
    assert result[0]["claims"] == []
    assert result[0]["busy"] is None
    assert resolved == set()


# ---- other_processes ------------------------------------------------------------

def test_other_processes_excludes_known_ollama_pids():
    def nvml_processes_fn():
        return [
            {"pid": 100, "size_mb": 500, "kind": "compute"},
            {"pid": 200, "size_mb": 300, "kind": "graphics"},
        ]

    result = core.other_processes(nvml_processes_fn, exclude_pids={100})
    assert result == [{"pid": 200, "size_mb": 300, "kind": "graphics"}]


# ---- pressure ---------------------------------------------------------------

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


# ---- combined_status full wiring -------------------------------------------------

def test_combined_status_full_wiring():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4),
              "expires_at": None}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 555}, busy_map={555: True})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda: snap,
        nvml_processes_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "kind": "compute"},
            {"pid": 999, "size_mb": 100, "kind": "graphics"},
        ],
    )
    entry = status["loaded"][0]
    assert entry["claims"] == [{"model": "m1", "owner": "x"}]
    assert entry["busy"] is True
    # pid 555 IS the "m1" runner -> excluded from other_processes; pid 999 stays.
    assert status["other_processes"] == [{"pid": 999, "size_mb": 100, "kind": "graphics"}]


def test_combined_status_other_processes_from_procinfo():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4), "expires_at": None}]
    snap = _snap(pid_map={"m1": 555})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda: snap,
        procinfo_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "name": "llama-server.exe", "cmdline": "...", "kind": "compute"},
            {"pid": 999, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
        ],
    )
    # pid 555 is the m1 runner -> excluded; 999 stays, WITH its size + name.
    assert status["other_processes"] == [
        {"pid": 999, "size_mb": 14492, "name": "python.exe",
         "cmdline": "python train.py", "kind": "compute"},
    ]


def test_combined_status_skips_snapshot_when_nothing_loaded():
    """An idle-status call must not pay the snapshot's subprocess/IO cost."""
    def exploding_snapshot():
        raise AssertionError("snapshot must not be captured for zero models")

    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama([]), snapshot_fn=exploding_snapshot,
    )
    assert status["loaded"] == []


# ---- ensure_free ------------------------------------------------------------

def test_ensure_free_already_free_short_circuits():
    ollama = FakeOllama([
        {"name": "m", "size_vram": gb_bytes(4), "expires_at": None},
    ])
    result = core.ensure_free(
        4, gpu_fn_const(8192), ollama, sleep=lambda *_: None
    )
    assert result["ok"] is True
    assert result["already_free"] is True
    assert result["unloaded"] == []
    assert ollama.unloaded == []  # nothing evicted


def test_ensure_free_unloads_largest_first_and_stops():
    models = [
        {"name": "small", "size_vram": gb_bytes(2), "expires_at": None},
        {"name": "huge", "size_vram": gb_bytes(10), "expires_at": None},
        {"name": "medium", "size_vram": gb_bytes(5), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    # free: 1000 (initial check) -> 11000 after first unload meets 8GB target.
    gpu_fn = gpu_fn_sequence([1000, 11000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)

    assert result["ok"] is True
    assert result["already_free"] is False
    # Largest-first: "huge" unloaded first; target met -> stop before others.
    assert ollama.unloaded == ["huge"]
    assert result["unloaded"] == ["huge"]
    assert result["free_mb"] == 11000
    assert result["target_mb"] == 8 * 1024


def test_ensure_free_unloads_multiple_until_target():
    models = [
        {"name": "a", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "b", "size_vram": gb_bytes(6), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    # 1000 initial, 5000 after first unload (still < 8GB), 9000 after second.
    gpu_fn = gpu_fn_sequence([1000, 5000, 9000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert result["ok"] is True
    assert ollama.unloaded == ["b", "a"]  # largest first
    assert result["free_mb"] == 9000


def test_ensure_free_cannot_reach_target():
    models = [
        {"name": "a", "size_vram": gb_bytes(2), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 3000])  # never reaches 8GB
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert result["ok"] is False
    assert ollama.unloaded == ["a"]
    assert result["free_mb"] == 3000


def test_ensure_free_unknown_vram_is_not_ok():
    ollama = FakeOllama([
        {"name": "a", "size_vram": gb_bytes(2), "expires_at": None},
    ])
    result = core.ensure_free(
        8, gpu_fn_const(None), ollama, sleep=lambda *_: None
    )
    assert result["ok"] is False
    assert result["free_mb"] is None


def test_ensure_free_settle_calls_sleep():
    calls = []
    models = [{"name": "a", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])
    core.ensure_free(
        8, gpu_fn, ollama, settle=0.5, sleep=lambda s: calls.append(s)
    )
    assert calls == [0.5]


# ---- is_protected -----------------------------------------------------------

def test_is_protected_true_when_active_claim_exists():
    snap = _snap(all_claims=[{"model": "m", "owner": "x"}])
    protected, detail = core.is_protected("m", snap)
    assert protected is True
    assert detail["claims"] == [{"model": "m", "owner": "x"}]


def test_is_protected_true_when_busy():
    snap = _snap(pid_map={"m": 123}, busy_map={123: True})
    protected, detail = core.is_protected("m", snap)
    assert protected is True
    assert detail["busy"] is True


def test_is_protected_false_when_unclaimed_and_idle():
    snap = _snap(pid_map={"m": 123}, busy_map={123: False})
    protected, _ = core.is_protected("m", snap)
    assert protected is False


def test_is_protected_false_when_no_pid_and_no_claim():
    snap = _snap(busy_map={123: True})   # a busy pid exists, but not for "m"
    protected, detail = core.is_protected("m", snap)
    assert protected is False
    assert detail["busy"] is None


# ---- ensure_free protection ---------------------------------------------------

def test_ensure_free_skips_protected_model_and_reports_declined():
    models = [
        {"name": "protected", "size_vram": gb_bytes(10), "expires_at": None},
        {"name": "free-game", "size_vram": gb_bytes(6), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 7000])

    result = core.ensure_free(
        6, gpu_fn, ollama, sleep=lambda *_: None,
        snapshot_fn=lambda: _snap(
            all_claims=[{"model": "protected", "owner": "other"}]),
    )
    assert ollama.unloaded == ["free-game"]  # "protected" skipped despite being largest
    assert result["ok"] is True
    assert len(result["declined"]) == 1
    assert result["declined"][0]["name"] == "protected"


def test_ensure_free_snapshots_once_for_the_whole_pass():
    """The coordination snapshot is captured ONCE per ensure_free call, not
    per candidate model — the old shape re-ran subprocess/NVML/file reads
    every loop iteration."""
    models = [
        {"name": "a", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "b", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "c", "size_vram": gb_bytes(4), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 2000, 3000, 13000])
    calls = {"n": 0}

    def snapshot_fn():
        calls["n"] += 1
        return _snap()

    core.ensure_free(12, gpu_fn, ollama, sleep=lambda *_: None,
                     snapshot_fn=snapshot_fn)
    assert calls["n"] == 1


def test_ensure_free_force_bypasses_protection_without_snapshotting():
    models = [{"name": "protected", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])

    def snapshot_fn():
        raise AssertionError("force=True must not pay for a snapshot")

    result = core.ensure_free(
        8, gpu_fn, ollama, sleep=lambda *_: None, force=True,
        snapshot_fn=snapshot_fn,
    )
    assert ollama.unloaded == ["protected"]
    assert result["declined"] == []


def test_ensure_free_protection_noop_when_snapshot_not_provided():
    """Existing callers that don't wire a snapshot see unchanged behavior."""
    models = [{"name": "a", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert ollama.unloaded == ["a"]
    assert result["declined"] == []


def test_ensure_free_skips_snapshot_when_no_models_to_evict():
    """Target unreachable with zero loaded models: no snapshot is captured."""
    def exploding_snapshot():
        raise AssertionError("snapshot must not be captured for zero models")

    result = core.ensure_free(
        8, gpu_fn_const(1000), FakeOllama([]), sleep=lambda *_: None,
        snapshot_fn=exploding_snapshot,
    )
    assert result["ok"] is False
    assert result["unloaded"] == []


def test_ensure_free_no_settle_sleep_when_unload_fails():
    """The settle sleep exists to let the driver release memory after an
    eviction — a FAILED unload released nothing, so sleeping is pure waste."""
    class RefusingOllama(FakeOllama):
        def unload(self, model):
            self.unloaded.append(model)
            return False

    models = [{"name": "a", "size_vram": gb_bytes(4), "expires_at": None}]
    ollama = RefusingOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 1000])
    sleeps = []
    core.ensure_free(8, gpu_fn, ollama, settle=0.5,
                     sleep=lambda s: sleeps.append(s))
    assert ollama.unloaded == ["a"]   # attempt made
    assert sleeps == []               # but no pointless settle wait


# ---- advise -----------------------------------------------------------------

def test_advise_recommends_max_loaded_when_many_and_low_free():
    models = [
        {"name": "a", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "b", "size_vram": gb_bytes(4), "expires_at": None},
    ]
    result = core.advise(gpu_fn_const(500), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_MAX_LOADED_MODELS=1" in joined


def test_advise_recommends_keep_alive_when_pinned_forever():
    models = [
        {"name": "pinned", "size_vram": gb_bytes(4),
         "expires_at": "0001-01-01T00:00:00Z"},
    ]
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_KEEP_ALIVE" in joined
    assert "pinned" in joined


def test_advise_quiet_when_healthy():
    models = [
        {"name": "a", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    assert result["suggestions"] == []


def test_advise_no_max_loaded_when_free_is_high():
    models = [
        {"name": "a", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
        {"name": "b", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    # Two models but plenty free -> no OLLAMA_MAX_LOADED_MODELS suggestion.
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_MAX_LOADED_MODELS" not in joined
