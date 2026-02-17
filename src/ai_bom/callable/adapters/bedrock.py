"""AWS Bedrock adapter for callable AI models."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters._base import BaseAdapter


class BedrockAdapter(BaseAdapter):
    """Adapter for AWS Bedrock converse API."""

    SDK_PACKAGE = "boto3"

    def _get_client(self) -> Any:
        if self._client is None:
            self._check_sdk()
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._kwargs.get("region_name", "us-east-1"),
            )
        return self._client

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        client = self._get_client()
        merged = {**self._kwargs, **kwargs}
        inference_config: dict[str, Any] = {}
        if "temperature" in merged:
            inference_config["temperature"] = merged["temperature"]
        if "max_tokens" in merged:
            inference_config["maxTokens"] = merged["max_tokens"]

        params: dict[str, Any] = {
            "modelId": self.model_name,
            "messages": [
                {"role": "user", "content": [{"text": prompt}]},
            ],
        }
        if inference_config:
            params["inferenceConfig"] = inference_config

        response = client.converse(**params)
        text = ""
        output = response.get("output", {})
        message = output.get("message", {})
        for block in message.get("content", []):
            if "text" in block:
                text += block["text"]

        usage_data = response.get("usage", {})
        usage = {}
        if usage_data:
            usage = {
                "input_tokens": usage_data.get("inputTokens", 0),
                "output_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": usage_data.get("totalTokens", 0),
            }
        return CallableResult(
            text=text,
            model_name=self.model_name,
            provider=self.provider,
            usage=usage,
            raw=response,
        )
