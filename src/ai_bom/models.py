"""Pydantic v2 models for AI-BOM scanner."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ComponentType(str, Enum):
    """Type of AI component detected."""

    llm_provider = "llm_provider"
    agent_framework = "agent_framework"
    model = "model"
    endpoint = "endpoint"
    container = "container"
    tool = "tool"
    mcp_server = "mcp_server"
    mcp_client = "mcp_client"
    workflow = "workflow"


class UsageType(str, Enum):
    """How the AI component is used."""

    completion = "completion"
    embedding = "embedding"
    image_gen = "image_gen"
    speech = "speech"
    agent = "agent"
    tool_use = "tool_use"
    orchestration = "orchestration"
    unknown = "unknown"


class Severity(str, Enum):
    """Risk severity levels."""

    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class SourceLocation(BaseModel):
    """Location where an AI component was detected."""

    file_path: str
    line_number: int | None = None
    context_snippet: str = ""


class RiskAssessment(BaseModel):
    """Risk assessment for an AI component."""

    score: int = Field(ge=0, le=100, default=0)
    severity: Severity = Severity.low
    factors: list[str] = Field(default_factory=list)
    owasp_categories: list[str] = Field(default_factory=list)


class AIComponent(BaseModel):
    """Detected AI component with metadata and risk assessment."""

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: ComponentType
    version: str = ""
    provider: str = ""
    model_name: str = ""
    location: SourceLocation
    usage_type: UsageType = UsageType.unknown
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    metadata: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    source: str = ""  # which scanner found this: "code", "docker", "network", "cloud", "n8n"


class N8nWorkflowInfo(BaseModel):
    """Information about an n8n workflow."""

    workflow_name: str
    workflow_id: str = ""
    nodes: list[str] = Field(default_factory=list)  # node type names
    connections: dict[str, list[str]] = Field(default_factory=dict)  # node -> connected nodes
    trigger_type: str = ""
    agent_chains: list[list[str]] = Field(default_factory=list)  # chains of agent node names


class ScanSummary(BaseModel):
    """Summary statistics for a scan."""

    total_components: int = 0
    total_files_scanned: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_provider: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    highest_risk_score: int = 0
    scan_duration_seconds: float = 0.0


class ScanResult(BaseModel):
    """Complete scan result with components, workflows, and summary."""

    target_path: str
    scan_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ai_bom_version: str = Field(default_factory=lambda: __import__("ai_bom").__version__)
    components: list[AIComponent] = Field(default_factory=list)
    n8n_workflows: list[N8nWorkflowInfo] = Field(default_factory=list)
    summary: ScanSummary = Field(default_factory=ScanSummary)

    def build_summary(self) -> None:
        """Populate summary from components."""
        self.summary.total_components = len(self.components)

        # Count by type
        for component in self.components:
            component_type = component.type.value
            self.summary.by_type[component_type] = self.summary.by_type.get(component_type, 0) + 1

        # Count by provider
        for component in self.components:
            if component.provider:
                self.summary.by_provider[component.provider] = (
                    self.summary.by_provider.get(component.provider, 0) + 1
                )

        # Count by severity
        for component in self.components:
            severity = component.risk.severity.value
            self.summary.by_severity[severity] = self.summary.by_severity.get(severity, 0) + 1

        # Count unique files scanned
        unique_files = set()
        for component in self.components:
            fp = component.location.file_path
            if fp and fp != "dependency files":
                unique_files.add(fp)
        self.summary.total_files_scanned = len(unique_files)

        # Find highest risk score
        if self.components:
            self.summary.highest_risk_score = max(
                component.risk.score for component in self.components
            )

    def to_cyclonedx(self) -> dict:
        """Generate CycloneDX 1.6 JSON-compatible dict."""
        # Map ComponentType to CycloneDX component types
        type_mapping = {
            ComponentType.llm_provider: "machine-learning-model",
            ComponentType.agent_framework: "framework",
            ComponentType.model: "machine-learning-model",
            ComponentType.endpoint: "service",
            ComponentType.container: "container",
            ComponentType.tool: "library",
            ComponentType.mcp_server: "service",
            ComponentType.mcp_client: "library",
            ComponentType.workflow: "framework",
        }

        # Build components list
        cdx_components = []
        for component in self.components:
            properties = []

            # Add risk score
            properties.append({"name": "trusera:risk_score", "value": str(component.risk.score)})

            # Add usage type
            properties.append({"name": "trusera:usage_type", "value": component.usage_type.value})

            # Add source location
            location_str = component.location.file_path
            if component.location.line_number:
                location_str += f":{component.location.line_number}"
            properties.append({"name": "trusera:source_location", "value": location_str})

            # Add provider if present
            if component.provider:
                properties.append({"name": "trusera:provider", "value": component.provider})

            # Add risk factors if present
            if component.risk.factors:
                properties.append(
                    {
                        "name": "trusera:risk_factors",
                        "value": ", ".join(component.risk.factors),
                    }
                )

            # Add flags if present
            if component.flags:
                properties.append({"name": "trusera:flags", "value": ", ".join(component.flags)})

            # Add model name if present
            if component.model_name:
                properties.append({"name": "trusera:model_name", "value": component.model_name})

            # Add source
            if component.source:
                properties.append({"name": "trusera:source", "value": component.source})

            # Build purl (Package URL) for interoperability
            purl = self._generate_purl(component)

            # Determine version field (omit if unknown)
            has_version = component.version and component.version != "unknown"
            version = component.version if has_version else None

            cdx_component = {
                "bom-ref": component.id,
                "type": type_mapping.get(component.type, "application"),
                "name": component.name,
                "description": f"{component.provider} {component.usage_type.value}".strip(),
                "properties": properties,
                "purl": purl,
            }

            # Only add version if it's valid
            if version:
                cdx_component["version"] = version

            cdx_components.append(cdx_component)

        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "version": 1,
            "serialNumber": f"urn:uuid:{uuid4()}",
            "metadata": {
                "timestamp": self.scan_timestamp,
                "tools": {
                    "components": [
                        {
                            "type": "application",
                            "name": "ai-bom",
                            "version": self.ai_bom_version,
                            "manufacturer": {
                                "name": "Trusera",
                                "url": ["https://trusera.dev"],
                            },
                        }
                    ]
                },
            },
            "components": cdx_components,
        }

    def _generate_purl(self, component: AIComponent) -> str:
        """Generate Package URL (purl) for a component.

        Args:
            component: The AI component to generate a purl for

        Returns:
            Package URL string in the format pkg:type/name@version
        """
        # Normalize package name: lowercase and use hyphens (PyPI convention)
        package_name = component.name.lower().replace("_", "-")

        # Determine purl type based on component metadata or source
        purl_type = "pypi"  # default

        # Check if it's a container
        if component.type == ComponentType.container:
            purl_type = "docker"
            # For Docker, extract image name from component name
            # Format is usually "image:tag" or just "image"
            if ":" in package_name:
                package_name, tag = package_name.split(":", 1)
                version = component.version or tag
            else:
                version = self._clean_version(component)
        # Check for npm packages (starts with @ or has /)
        elif package_name.startswith("@") or "/" in package_name:
            purl_type = "npm"
            version = self._clean_version(component)
        # Default to PyPI
        else:
            version = self._clean_version(component)

        # Build purl
        if version:
            return f"pkg:{purl_type}/{package_name}@{version}"
        return f"pkg:{purl_type}/{package_name}"

    @staticmethod
    def _clean_version(component: AIComponent) -> str:
        """Return version string or empty if unknown."""
        v = component.version
        if v and v != "unknown":
            return v
        return ""
