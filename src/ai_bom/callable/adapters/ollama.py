"""Ollama adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class OllamaAdapter(BaseAdapter):
    """Adapter for Ollama local inference API."""

    SDK_PACKAGE = "ollama"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import ollama

            host = self._kwargs.get("host")
            if host:
                self._client = ollama.Client(host=host)
            else:
                self._client = ollama.Client()
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        options: dict[str, Any] = {}
        if "temperature" in merged:
            options["temperature"] = merged["temperature"]

        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options=options or None,
        )
        usage = {}
        if "prompt_eval_count" in response:
            usage["prompt_tokens"] = response["prompt_eval_count"]
        if "eval_count" in response:
            usage["completion_tokens"] = response["eval_count"]
        return CallableResult(
            text=response["message"]["content"],
            model_name=self.model_name,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
