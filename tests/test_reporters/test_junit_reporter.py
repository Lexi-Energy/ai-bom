"""Tests for JUnit XML reporter."""

import xml.etree.ElementTree as ET

from ai_bom.models import (
    AIComponent,
    ComponentType,
    RiskAssessment,
    ScanResult,
    Severity,
    SourceLocation,
)
from ai_bom.reporters.junit_reporter import JUnitReporter


class TestJUnitReporter:
    """Test cases for JUnit XML reporter."""

    def test_renders_valid_xml(self, multi_component_result):
        """Test that output is valid XML."""
        reporter = JUnitReporter()
        output = reporter.render(multi_component_result)

        # Parse to ensure valid XML
        root = ET.fromstring(output)  # noqa: S314
        assert root.tag == "testsuite"

    def test_testsuite_attributes(self, multi_component_result):
        """Test that testsuite has correct attributes."""
        reporter = JUnitReporter()
        output = reporter.render(multi_component_result)

        root = ET.fromstring(output)  # noqa: S314

        assert "name" in root.attrib
        assert "tests" in root.attrib
        assert "failures" in root.attrib
        assert "timestamp" in root.attrib

        # Tests count should match components
        assert root.attrib["tests"] == str(len(multi_component_result.components))

    def test_testcases_created(self, multi_component_result):
        """Test that testcase elements are created for each component."""
        reporter = JUnitReporter()
        output = reporter.render(multi_component_result)

        root = ET.fromstring(output)  # noqa: S314
        testcases = root.findall("testcase")

        assert len(testcases) == len(multi_component_result.components)

    def test_critical_severity_is_failure(self):
        """Test that critical severity components are marked as failures."""
        result = ScanResult(target_path="/test")
        result.components.append(
            AIComponent(
                name="Critical Component",
                type=ComponentType.llm_provider,
                provider="Test",
                location=SourceLocation(file_path="/test/file.py"),
                risk=RiskAssessment(score=90, severity=Severity.critical),
                source="code",
            )
        )
        result.build_summary()

        reporter = JUnitReporter()
        output = reporter.render(result)

        root = ET.fromstring(output)  # noqa: S314
        testcase = root.find("testcase")
        failure = testcase.find("failure")

        assert failure is not None
        assert failure.attrib["type"] == "critical"

    def test_high_severity_is_failure(self):
        """Test that high severity components are marked as failures."""
        result = ScanResult(target_path="/test")
        result.components.append(
            AIComponent(
                name="High Risk Component",
                type=ComponentType.llm_provider,
                provider="Test",
                location=SourceLocation(file_path="/test/file.py"),
                risk=RiskAssessment(score=70, severity=Severity.high),
                source="code",
            )
        )
        result.build_summary()

        reporter = JUnitReporter()
        output = reporter.render(result)

        root = ET.fromstring(output)  # noqa: S314
        failures_count = int(root.attrib["failures"])

        assert failures_count == 1

    def test_hardcoded_api_key_is_failure(self):
        """Test that hardcoded_api_key flag marks component as failure."""
        result = ScanResult(target_path="/test")
        result.components.append(
            AIComponent(
                name="Component with API Key",
                type=ComponentType.llm_provider,
                provider="Test",
                location=SourceLocation(file_path="/test/file.py"),
                flags=["hardcoded_api_key"],
                risk=RiskAssessment(score=50, severity=Severity.medium),
                source="code",
            )
        )
        result.build_summary()

        reporter = JUnitReporter()
        output = reporter.render(result)

        root = ET.fromstring(output)  # noqa: S314
        testcase = root.find("testcase")
        failure = testcase.find("failure")

        assert failure is not None

    def test_shadow_ai_is_failure(self):
        """Test that shadow_ai flag marks component as failure."""
        result = ScanResult(target_path="/test")
        result.components.append(
            AIComponent(
                name="Shadow AI Component",
                type=ComponentType.llm_provider,
                provider="Test",
                location=SourceLocation(file_path="/test/file.py"),
                flags=["shadow_ai"],
                risk=RiskAssessment(score=50, severity=Severity.medium),
                source="code",
            )
        )
        result.build_summary()

        reporter = JUnitReporter()
        output = reporter.render(result)

        root = ET.fromstring(output)  # noqa: S314
        failures_count = int(root.attrib["failures"])

        assert failures_count == 1

    def test_low_severity_not_failure(self):
        """Test that low severity components without security flags are not failures."""
        result = ScanResult(target_path="/test")
        result.components.append(
            AIComponent(
                name="Low Risk Component",
                type=ComponentType.llm_provider,
                provider="Test",
                location=SourceLocation(file_path="/test/file.py"),
                risk=RiskAssessment(score=20, severity=Severity.low),
                source="code",
            )
        )
        result.build_summary()

        reporter = JUnitReporter()
        output = reporter.render(result)

        root = ET.fromstring(output)  # noqa: S314
        failures_count = int(root.attrib["failures"])

        assert failures_count == 0

    def test_properties_included(self, multi_component_result):
        """Test that properties element is included with metadata."""
        reporter = JUnitReporter()
        output = reporter.render(multi_component_result)

        root = ET.fromstring(output)  # noqa: S314
        properties = root.find("properties")

        assert properties is not None

        # Check for specific properties
        prop_names = [prop.attrib["name"] for prop in properties.findall("property")]
        assert "ai_bom_version" in prop_names
        assert "highest_risk_score" in prop_names

    def test_system_out_included(self, multi_component_result):
        """Test that system-out element is included with component metadata."""
        reporter = JUnitReporter()
        output = reporter.render(multi_component_result)

        root = ET.fromstring(output)  # noqa: S314
        testcase = root.find("testcase")
        system_out = testcase.find("system-out")

        assert system_out is not None
        assert len(system_out.text) > 0

    def test_write_to_file(self, multi_component_result, tmp_path):
        """Test writing JUnit XML to file."""
        reporter = JUnitReporter()
        path = tmp_path / "junit.xml"
        reporter.write(multi_component_result, path)

        assert path.exists()
        content = path.read_text()

        # Verify it's valid XML
        root = ET.fromstring(content)  # noqa: S314
        assert root.tag == "testsuite"
