"""JUnit XML reporter for AI-BOM scan results."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET

from ai_bom.models import AIComponent, ScanResult
from ai_bom.reporters.base import BaseReporter


class JUnitReporter(BaseReporter):
    """Reporter that generates JUnit XML output.

    Maps scan results to JUnit XML format:
    - Overall scan is a test suite
    - Each component is a test case
    - Components with severity >= high or specific flags become failures
    """

    def render(self, result: ScanResult) -> str:
        """Render scan result as JUnit XML.

        Args:
            result: The scan result to render

        Returns:
            JUnit XML formatted string
        """
        # Create root testsuite element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", f"AI-BOM Scan: {result.target_path}")
        testsuite.set("tests", str(result.summary.total_components))
        testsuite.set("timestamp", result.scan_timestamp)

        # Count failures
        failures = 0
        for component in result.components:
            if self._is_failure(component):
                failures += 1

        testsuite.set("failures", str(failures))
        testsuite.set("errors", "0")
        testsuite.set("time", str(result.summary.scan_duration_seconds))

        # Add properties
        properties = ET.SubElement(testsuite, "properties")

        # Add scan metadata as properties
        prop = ET.SubElement(properties, "property")
        prop.set("name", "ai_bom_version")
        prop.set("value", result.ai_bom_version)

        prop = ET.SubElement(properties, "property")
        prop.set("name", "highest_risk_score")
        prop.set("value", str(result.summary.highest_risk_score))

        prop = ET.SubElement(properties, "property")
        prop.set("name", "total_files_scanned")
        prop.set("value", str(result.summary.total_files_scanned))

        # Add each component as a testcase
        for component in result.components:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", component.name)
            testcase.set("classname", f"{component.type.value}.{component.provider}")
            testcase.set("time", "0")

            # Check if component should be marked as failure
            if self._is_failure(component):
                failure = ET.SubElement(testcase, "failure")
                failure.set("type", component.risk.severity.value)
                failure.set(
                    "message",
                    f"{component.name} has {component.risk.severity.value} severity "
                    f"(risk score: {component.risk.score})",
                )

                # Build failure details
                details = []
                details.append(f"Component: {component.name}")
                details.append(f"Type: {component.type.value}")
                details.append(f"Provider: {component.provider}")
                details.append(f"Risk Score: {component.risk.score}/100")
                details.append(f"Severity: {component.risk.severity.value}")
                details.append(f"Location: {component.location.file_path}")

                if component.location.line_number:
                    details.append(f"Line: {component.location.line_number}")

                if component.model_name:
                    details.append(f"Model: {component.model_name}")

                if component.flags:
                    details.append(f"Flags: {', '.join(component.flags)}")

                if component.risk.factors:
                    details.append(f"Risk Factors: {', '.join(component.risk.factors)}")

                failure.text = "\n".join(details)

            # Add system-out with component metadata
            system_out = ET.SubElement(testcase, "system-out")
            out_lines = []
            out_lines.append(f"Provider: {component.provider}")
            out_lines.append(f"Type: {component.type.value}")
            out_lines.append(f"Usage: {component.usage_type.value}")
            out_lines.append(f"Risk Score: {component.risk.score}")
            out_lines.append(f"Source: {component.source}")

            if component.location.context_snippet:
                out_lines.append(f"Context: {component.location.context_snippet}")

            system_out.text = "\n".join(out_lines)

        # Convert to string with XML declaration
        tree = ET.ElementTree(testsuite)
        output = io.BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue().decode("utf-8")

    def _is_failure(self, component: AIComponent) -> bool:
        """Determine if a component should be marked as a test failure.

        Args:
            component: AIComponent to check

        Returns:
            True if component should be marked as failure
        """
        # High and critical severity are failures
        # Specific security flags are failures
        security_flags = {
            "hardcoded_api_key",
            "shadow_ai",
            "webhook_no_auth",
            "code_http_tools",
            "mcp_unknown_server",
            "multi_agent_no_trust",
            "no_auth",
        }

        return component.risk.severity.value in ("high", "critical") or any(
            flag in security_flags for flag in component.flags
        )
