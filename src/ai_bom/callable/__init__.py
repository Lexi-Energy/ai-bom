"""Callable Models — turn AI-BOM scan results into invocable LLM wrappers.

Usage::

    from ai_bom.callable import get_callables, CallableModel, CallableResult

    callables = get_callables(scan_result, api_key="sk-...")
    for model in callables:
        result = model("Is this input safe?")
        print(result.text)
"""

from __future__ import annotations

from ai_bom.callable._factory import (
    create_callable,
    get_callables,
    get_callables_from_cdx,
)
from ai_bom.callable._protocol import CallableModel, CallableResult

__all__ = [
    "CallableModel",
    "CallableResult",
    "create_callable",
    "get_callables",
    "get_callables_from_cdx",
]
