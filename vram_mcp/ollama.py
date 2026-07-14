"""Minimal Ollama HTTP client.

Pure module: no ``mcp`` import. Every method swallows transport errors and
returns a benign default (``[]`` / ``False``) so callers never have to guard
against a down Ollama daemon.
"""

from __future__ import annotations

from typing import Optional

import requests


class OllamaClient:
    """Thin client over the Ollama REST API used for VRAM management.

    Only the endpoints needed to see and evict resident models are wired up:
    ``/api/ps`` (list loaded), and ``/api/generate`` with a ``keep_alive`` of
    ``0`` (unload) or a duration (warm/pin).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ps(self) -> list[dict]:
        """Return the list of currently loaded models (``/api/ps``).

        Each model has at least ``name``, ``size`` (bytes), ``size_vram``
        (bytes) and ``expires_at``. Returns ``[]`` on any transport error.
        """
        try:
            resp = self.session.get(self._url("/api/ps"), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []
        models = data.get("models")
        return models if isinstance(models, list) else []

    def _generate(self, model: str, keep_alive) -> bool:
        """POST an empty generate with a ``keep_alive`` to load/unload a model."""
        payload = {
            "model": model,
            "prompt": "",
            "keep_alive": keep_alive,
            "stream": False,
        }
        try:
            resp = self.session.post(
                self._url("/api/generate"),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException:
            return False
        return 200 <= resp.status_code < 300

    def unload(self, model: str) -> bool:
        """Evict ``model`` from VRAM now (``keep_alive=0``). True on 2xx."""
        return self._generate(model, 0)

    def warm(self, model: str, keep_alive: str = "5m") -> bool:
        """Load/pin ``model`` for ``keep_alive`` (e.g. ``"5m"``). True on 2xx."""
        return self._generate(model, keep_alive)
