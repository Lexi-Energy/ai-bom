"""Mistral adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class MistralAdapter(BaseAdapter):
    """Adapter for Mistral chat completions API."""

    SDK_PACKAGE = "mistralai"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            from mistralai import Mistral

            self._client = Mistral(
                api_key=self._kwargs.get("api_key"),
            )
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        response = client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=merged.get("temperature", 0.0),
            max_tokens=merged.get("max_tokens"),
        )
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return CallableResult(
            text=text,
            model_name=response.model or self.model_name,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
