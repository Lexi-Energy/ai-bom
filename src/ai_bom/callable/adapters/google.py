"""Google Generative AI adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class GoogleAdapter(BaseAdapter):
    """Adapter for Google Generative AI (Gemini) API."""

    SDK_PACKAGE = "google.generativeai"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import google.generativeai as genai

            api_key = self._kwargs.get("api_key")
            if api_key:
                genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model_name)
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        model = self._get_client()
        merged = {**self._kwargs, **kwargs}
        generation_config: dict[str, Any] = {}
        if "temperature" in merged:
            generation_config["temperature"] = merged["temperature"]
        if "max_tokens" in merged:
            generation_config["max_output_tokens"] = merged["max_tokens"]

        response = model.generate_content(
            prompt,
            generation_config=generation_config or None,
        )
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", 0),
                "completion_tokens": getattr(meta, "candidates_token_count", 0),
                "total_tokens": getattr(meta, "total_token_count", 0),
            }
        return CallableResult(
            text=response.text,
            model_name=self.model_name,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
