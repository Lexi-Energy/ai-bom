"""Protocol and result types for callable AI models."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class CallableResult(BaseModel):
    """Result from calling an AI model."""

    text: str
    model_name: str = ""
    provider: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: Any = None


@runtime_checkable
class CallableModel(Protocol):
    """Protocol for callable AI model wrappers.

    Downstream tools (e.g. Giskard) can type-check against this
    without importing ai-bom:

        if isinstance(obj, CallableModel):
            result = obj("Is this safe?")
    """

    model_name: str
    provider: str

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult: ...
