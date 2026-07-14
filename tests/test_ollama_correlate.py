"""Tests for vram_mcp.ollama_correlate — pure except real tmp_path manifests."""

import json

from vram_mcp import ollama_correlate as oc

_REAL_DIGEST = "a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"


def _write_manifest(manifests_root, registry_host, namespace, name, tag, model_digest):
    """Write a manifest at the REAL Ollama on-disk depth:
    <manifests_root>/<registry_host>/<namespace>/<name>/<tag>
    (verified live: registry.ollama.ai/library/gemma3/27b)."""
    path = manifests_root / registry_host / namespace / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": f"sha256:{model_digest}"},
            {"mediaType": "application/vnd.ollama.image.template",
             "digest": "sha256:deadbeef"},
        ],
    }))


def test_extract_model_digest_from_real_cmdline():
    cmdline = (
        r"C:\Ollama\llama-server.exe --model "
        rf"C:\Users\x\.ollama\models\blobs\sha256-{_REAL_DIGEST} --port 12288"
    )
    assert oc._extract_model_digest(cmdline) == _REAL_DIGEST


def test_extract_model_digest_missing_flag_returns_none():
    assert oc._extract_model_digest("llama-server.exe --port 12288") is None


def test_extract_model_digest_short_hash_not_matched():
    # A malformed/truncated digest (not 64 hex chars) is never extracted --
    # never guess a partial match.
    assert oc._extract_model_digest("llama-server.exe --model /x/sha256-abc123") is None


def test_resolve_tag_for_digest_library_namespace(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", "abc123")
    assert oc._resolve_tag_for_digest("abc123", manifests) == "qwen3:8b"


def test_resolve_tag_for_digest_non_library_namespace(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "someuser", "mymodel", "latest", "abc123")
    assert oc._resolve_tag_for_digest("abc123", manifests) == "someuser/mymodel:latest"


def test_resolve_tag_for_digest_no_match_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", "abc123")
    assert oc._resolve_tag_for_digest("nonexistent", manifests) is None


def test_resolve_tag_for_digest_missing_dir_returns_none(tmp_path):
    assert oc._resolve_tag_for_digest("abc123", tmp_path / "nope") is None


def test_resolve_tag_for_digest_wrong_depth_skipped(tmp_path):
    # A manifest-shaped file at the WRONG depth (3 levels, missing the
    # registry-host segment) is skipped, not guessed at.
    manifests = tmp_path / "manifests"
    path = manifests / "library" / "qwen3" / "8b"  # only 3 levels deep
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "layers": [{"mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:abc123"}],
    }))
    assert oc._resolve_tag_for_digest("abc123", manifests) is None


def test_find_pid_for_model_matches_correlated_tag(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 43176, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} --port 1"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid == 43176


def test_find_pid_for_model_no_matching_process_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)
    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=lambda: [], manifests_root=manifests
    )
    assert pid is None


def test_find_pid_for_model_digest_mismatch_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)
    other_digest = "b" * 64

    def fake_list_processes():
        return [{"pid": 1, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{other_digest}"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid is None
