"""Tests for vram_mcp.ollama — pure, mocked session, no real Ollama."""

import requests

from vram_mcp.ollama import OllamaClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


class FakeSession:
    """Records GET/POST calls and returns queued responses."""

    def __init__(self, get_response=None, post_response=None, raise_on=None):
        self.get_response = get_response
        self.post_response = post_response
        self.raise_on = raise_on  # "get" | "post" | None
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout=None):
        self.get_calls.append({"url": url, "timeout": timeout})
        if self.raise_on == "get":
            raise requests.ConnectionError("refused")
        return self.get_response

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        if self.raise_on == "post":
            raise requests.ConnectionError("refused")
        return self.post_response


def test_ps_parses_models():
    models = [
        {"name": "llama3", "size": 5000000000, "size_vram": 4800000000,
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    session = FakeSession(get_response=FakeResponse(200, {"models": models}))
    client = OllamaClient(session=session)
    assert client.ps() == models
    assert session.get_calls[0]["url"] == "http://127.0.0.1:11434/api/ps"


def test_ps_missing_models_key_returns_empty():
    session = FakeSession(get_response=FakeResponse(200, {}))
    assert OllamaClient(session=session).ps() == []


def test_ps_transport_error_returns_empty():
    session = FakeSession(raise_on="get")
    assert OllamaClient(session=session).ps() == []


def test_ps_http_error_returns_empty():
    session = FakeSession(get_response=FakeResponse(500, {}))
    assert OllamaClient(session=session).ps() == []


def test_tags_returns_model_sizes():
    payload = {"models": [
        {"name": "llama3:8b", "size": 4 * 1024 * 1024 * 1024},
        {"name": "qwen3:32b", "size": 20 * 1024 * 1024 * 1024},
    ]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    client = OllamaClient(session=session)
    tags = client.tags()
    assert tags["llama3:8b"] == 4096
    assert tags["qwen3:32b"] == 20480
    assert session.get_calls[0]["url"] == "http://127.0.0.1:11434/api/tags"


def test_tags_transport_error_returns_empty():
    session = FakeSession(raise_on="get")
    assert OllamaClient(session=session).tags() == {}


def test_tags_http_error_returns_empty():
    session = FakeSession(get_response=FakeResponse(500, {}))
    assert OllamaClient(session=session).tags() == {}


def test_tags_missing_models_key_returns_empty():
    session = FakeSession(get_response=FakeResponse(200, {}))
    assert OllamaClient(session=session).tags() == {}


def test_tags_skips_nameless_rows():
    payload = {"models": [{"size": 1024}, {"name": "a", "size": 1048576}]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    assert OllamaClient(session=session).tags() == {"a": 1}


def test_tags_skips_unknown_sizes_and_non_dict_rows():
    payload = {"models": [
        "not-a-dict",
        {"name": "nosize"},
        {"name": "garbage", "size": "big"},
        {"name": "ok", "size": 2 * 1024 * 1024},
    ]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    assert OllamaClient(session=session).tags() == {"ok": 2}


def test_unload_posts_keep_alive_zero_and_returns_true():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.unload("llama3") is True

    call = session.post_calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/generate"
    assert call["json"]["model"] == "llama3"
    assert call["json"]["keep_alive"] == 0
    assert call["json"]["prompt"] == ""
    assert call["json"]["stream"] is False


def test_warm_posts_keep_alive_duration_and_returns_true():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.warm("llama3") is True
    assert session.post_calls[0]["json"]["keep_alive"] == "5m"


def test_warm_custom_keep_alive():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.warm("llama3", keep_alive="1h") is True
    assert session.post_calls[0]["json"]["keep_alive"] == "1h"


def test_unload_non_2xx_returns_false():
    session = FakeSession(post_response=FakeResponse(404))
    assert OllamaClient(session=session).unload("nope") is False


def test_warm_transport_error_returns_false():
    session = FakeSession(raise_on="post")
    assert OllamaClient(session=session).warm("llama3") is False


def test_base_url_trailing_slash_stripped():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(base_url="http://host:11434/", session=session)
    client.unload("m")
    assert session.post_calls[0]["url"] == "http://host:11434/api/generate"
