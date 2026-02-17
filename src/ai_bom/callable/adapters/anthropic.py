"""Anthropic adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic messages API."""

    SDK_PACKAGE = "anthropic"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._kwargs.get("api_key"),
            )
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        response = client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=merged.get("max_tokens", 1024),
            temperature=merged.get("temperature", 0.0),
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        return CallableResult(
            text=text,
            model_name=response.model,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
