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


# ---- attach_claims ------------------------------------------------------------

def test_attach_claims_adds_claims_list_per_model():
    loaded = [{"name": "m1"}, {"name": "m2"}]

    def list_claims_fn(model):
        return [{"owner": "x"}] if model == "m1" else []

    result = core.attach_claims(loaded, list_claims_fn)
    assert result[0]["claims"] == [{"owner": "x"}]
    assert result[1]["claims"] == []


# ---- attach_busy ----------------------------------------------------------------

def test_attach_busy_resolves_pid_and_busy_flag():
    loaded = [{"name": "m1"}, {"name": "m2"}]

    def find_pid_fn(model):
        return 123 if model == "m1" else None

    def busy_fn(pid):
        return pid == 123

    result, resolved = core.attach_busy(loaded, find_pid_fn, busy_fn)
    assert result[0]["busy"] is True
    assert result[1]["busy"] is None  # no pid -> never call busy_fn's real signal
    assert resolved == {123}


# ---- other_processes ------------------------------------------------------------

def test_other_processes_excludes_known_ollama_pids():
    def nvml_processes_fn():
        return [
            {"pid": 100, "size_mb": 500, "kind": "compute"},
            {"pid": 200, "size_mb": 300, "kind": "graphics"},
        ]

    result = core.other_processes(nvml_processes_fn, exclude_pids={100})
    assert result == [{"pid": 200, "size_mb": 300, "kind": "graphics"}]


# ---- combined_status full wiring -------------------------------------------------

def test_combined_status_full_wiring():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4),
              "expires_at": None}]
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        list_claims_fn=lambda model: [{"owner": "x"}],
        find_pid_fn=lambda model: 555,
        busy_fn=lambda pid: True,
        nvml_processes_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "kind": "compute"},
            {"pid": 999, "size_mb": 100, "kind": "graphics"},
        ],
    )
    entry = status["loaded"][0]
    assert entry["claims"] == [{"owner": "x"}]
    assert entry["busy"] is True
    # pid 555 IS the "m1" runner -> excluded from other_processes; pid 999 stays.
    assert status["other_processes"] == [{"pid": 999, "size_mb": 100, "kind": "graphics"}]


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
