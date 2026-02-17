"""OpenAI adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI chat completions API."""

    SDK_PACKAGE = "openai"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import openai

            self._client = openai.OpenAI(
                api_key=self._kwargs.get("api_key"),
                base_url=self._kwargs.get("base_url"),
            )
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=merged.get("temperature", 0.0),
            max_tokens=merged.get("max_tokens"),
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return CallableResult(
            text=choice.message.content or "",
            model_name=response.model,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
