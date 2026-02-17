"""Factory functions for creating callable AI model wrappers."""

from __future__ import annotations

from typing import Any

from ai_bom.callable._protocol import CallableResult
from ai_bom.callable.adapters import get_adapter_class
from ai_bom.callable.adapters._base import BaseAdapter
from ai_bom.models import AIComponent, ComponentType, ScanResult


def create_callable(
    component: AIComponent,
    **kwargs: Any,
) -> BaseAdapter:
    """Create a callable model wrapper from an AIComponent.

    Args:
        component: An AI component from a scan result.
        **kwargs: Extra arguments forwarded to the adapter (api_key, base_url, etc.).

    Returns:
        A callable adapter instance.

    Raises:
        KeyError: If the component's provider is not supported.
        ValueError: If the component has no provider set.
    """
    provider = component.provider.lower()
    if not provider:
        msg = f"Component {component.name!r} has no provider set"
        raise ValueError(msg)

    model_name = component.model_name or component.name
    cls = get_adapter_class(provider)
    return cls(model_name=model_name, provider=provider, **kwargs)


def get_callables(
    source: ScanResult | list[AIComponent],
    **kwargs: Any,
) -> list[BaseAdapter]:
    """Create callable wrappers for all LLM/model components in a scan result.

    Skips components that are not LLM providers or models, and components
    with unsupported or missing providers.

    Args:
        source: A ScanResult or a list of AIComponent objects.
        **kwargs: Extra arguments forwarded to each adapter.

    Returns:
        List of callable adapter instances.
    """
    components = source.components if isinstance(source, ScanResult) else source

    callables: list[BaseAdapter] = []
    callable_types = {ComponentType.llm_provider, ComponentType.model}
    for component in components:
        if component.type not in callable_types:
            continue
        if not component.provider:
            continue
        try:
            callables.append(create_callable(component, **kwargs))
        except (KeyError, ValueError):
            continue
    return callables


def get_callables_from_cdx(
    cdx: dict[str, Any],
    **kwargs: Any,
) -> list[BaseAdapter]:
    """Create callable wrappers from a CycloneDX JSON dict.

    Parses ``trusera:*`` properties to reconstruct provider and model info,
    then builds adapters for each machine-learning-model component.

    Args:
        cdx: A CycloneDX 1.6 JSON dict (as produced by ``ScanResult.to_cyclonedx()``).
        **kwargs: Extra arguments forwarded to each adapter.

    Returns:
        List of callable adapter instances.
    """
    callables: list[BaseAdapter] = []
    for comp in cdx.get("components", []):
        comp_type = comp.get("type", "")
        if comp_type not in ("machine-learning-model", "framework", "library"):
            continue

        props = {p["name"]: p["value"] for p in comp.get("properties", [])}
        provider = props.get("trusera:provider", "")
        if not provider:
            continue

        model_name = props.get("trusera:model_name", "") or comp.get("name", "")
        try:
            cls = get_adapter_class(provider)
            callables.append(cls(model_name=model_name, provider=provider, **kwargs))
        except KeyError:
            continue
    return callables


__all__ = ["CallableResult", "create_callable", "get_callables", "get_callables_from_cdx"]
