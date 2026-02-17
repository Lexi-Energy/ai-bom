"""Adapter registry for callable AI models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_bom.callable.adapters._base import BaseAdapter

# Provider name -> (module path, class name)
# Lazy-loaded to avoid importing SDKs at module level.
_ADAPTER_MAP: dict[str, tuple[str, str]] = {
    "openai": ("ai_bom.callable.adapters.openai", "OpenAIAdapter"),
    "anthropic": ("ai_bom.callable.adapters.anthropic", "AnthropicAdapter"),
    "google": ("ai_bom.callable.adapters.google", "GoogleAdapter"),
    "bedrock": ("ai_bom.callable.adapters.bedrock", "BedrockAdapter"),
    "aws": ("ai_bom.callable.adapters.bedrock", "BedrockAdapter"),
    "ollama": ("ai_bom.callable.adapters.ollama", "OllamaAdapter"),
    "mistral": ("ai_bom.callable.adapters.mistral", "MistralAdapter"),
    "cohere": ("ai_bom.callable.adapters.cohere", "CohereAdapter"),
}

ADAPTERS: dict[str, str] = {k: v[1] for k, v in _ADAPTER_MAP.items()}


def get_adapter_class(provider: str) -> type[BaseAdapter]:
    """Get the adapter class for a provider name.

    Args:
        provider: Provider name (e.g. "openai", "anthropic").

    Returns:
        The adapter class.

    Raises:
        KeyError: If the provider is not supported.
    """
    key = provider.lower()
    if key not in _ADAPTER_MAP:
        supported = ", ".join(sorted({k for k in _ADAPTER_MAP if k != "aws"}))
        msg = f"Unsupported provider {provider!r}. Supported: {supported}"
        raise KeyError(msg)
    module_path, class_name = _ADAPTER_MAP[key]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


__all__ = ["ADAPTERS", "get_adapter_class"]
