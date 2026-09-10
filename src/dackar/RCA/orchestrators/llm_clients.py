"""
llm_clients — LLM client Protocol and concrete implementations.

Extracted from rca_reasoning_orchestrator.py.  The parent module re-exports
all three names for backward-compatible imports.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Protocol

import requests

JsonDict = Dict[str, Any]


class LLMClient(Protocol):
    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        ...


class DummyLLMClient:
    """
    Development-only LLM client used by build_dev_orchestrator().

    Intentionally raises so the real synthesizer falls back to the
    deterministic template path.  This lets the pipeline run end-to-end
    before Ollama or another real LLM backend is wired in.
    """

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> JsonDict:
        raise RuntimeError(
            "DummyLLMClient intentionally forces fallback synthesis in local development."
        )


class OllamaLLMClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama response was not valid JSON: {exc}") from exc
