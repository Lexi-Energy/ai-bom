"""Base adapter for callable AI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai_bom.callable._protocol import CallableResult


class BaseAdapter(ABC):
    """Abstract base class for provider-specific model adapters."""

    SDK_PACKAGE: str = ""

    def __init__(self, model_name: str, provider: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.provider = provider
        self._kwargs = kwargs
        self._client: Any = None

    def _check_sdk(self) -> None:
        """Raise ImportError with a helpful message if the SDK is not installed."""
        if not self.SDK_PACKAGE:
            return
        try:
            __import__(self.SDK_PACKAGE)
        except ImportError:
            extra = f"callable-{self.provider}"
            msg = (
                f"{self.SDK_PACKAGE!r} is required for {self.provider} models. "
                f"Install it with: pip install 'ai-bom[{extra}]'"
            )
            raise ImportError(msg) from None

    @abstractmethod
    def _get_client(self) -> Any:
        """Create or return the provider SDK client."""
        ...

    @abstractmethod
    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        """Call the model with a prompt and return a structured result."""
        ...
