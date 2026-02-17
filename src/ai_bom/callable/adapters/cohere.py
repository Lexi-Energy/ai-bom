"""Cohere adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class CohereAdapter(BaseAdapter):
    """Adapter for Cohere chat API."""

    SDK_PACKAGE = "cohere"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import cohere

            self._client = cohere.ClientV2(
                api_key=self._kwargs.get("api_key"),
            )
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in merged:
            params["temperature"] = merged["temperature"]
        if "max_tokens" in merged:
            params["max_tokens"] = merged["max_tokens"]

        response = client.chat(**params)
        text = ""
        if response.message and response.message.content:
            for block in response.message.content:
                if hasattr(block, "text"):
                    text += block.text
        usage = {}
        if response.usage and response.usage.tokens:
            tokens = response.usage.tokens
            usage = {
                "input_tokens": getattr(tokens, "input_tokens", 0),
                "output_tokens": getattr(tokens, "output_tokens", 0),
            }
        return CallableResult(
            text=text,
            model_name=self.model_name,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
