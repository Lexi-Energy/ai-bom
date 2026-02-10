"""Tests for the policy enforcement engine."""

from __future__ import annotations

import pytest

from ai_bom.models import (
    AIComponent,
    ComponentType,
    RiskAssessment,
    ScanResult,
    Severity,
    SourceLocation,
    UsageType,
)
from ai_bom.policy import Policy, evaluate_policy, load_policy


def _make_component(
    name: str = "test",
    provider: str = "TestProvider",
    severity: Severity = Severity.low,
    score: int = 10,
    flags: list[str] | None = None,
) -> AIComponent:
    """Helper to create a test component."""
    return AIComponent(
        name=name,
        type=ComponentType.llm_provider,
        provider=provider,
        location=SourceLocation(file_path="test.py", line_number=1),
        usage_type=UsageType.completion,
        risk=RiskAssessment(score=score, severity=severity, factors=[]),
        flags=flags or [],
        source="code",
    )


def _make_result(components: list[AIComponent] | None = None) -> ScanResult:
    """Helper to create a scan result."""
    result = ScanResult(target_path="/test")
    if components:
        result.components = components
    result.build_summary()
    return result


class TestPolicyModel:
    def test_default_policy(self):
        policy = Policy()
        assert policy.max_critical is None
        assert policy.max_high is None
        assert policy.max_risk_score is None
        assert policy.block_providers == []
        assert policy.block_flags == []

    def test_policy_with_values(self):
        policy = Policy(
            max_critical=0,
            max_high=5,
            max_risk_score=75,
            block_providers=["Ollama"],
            block_flags=["hardcoded_api_key"],
        )
        assert policy.max_critical == 0
        assert policy.max_high == 5
        assert policy.max_risk_score == 75
        assert policy.block_providers == ["Ollama"]
        assert policy.block_flags == ["hardcoded_api_key"]


class TestLoadPolicy:
    def test_load_valid_policy(self, tmp_path):
        policy_file = tmp_path / "policy.yml"
        policy_file.write_text("max_critical: 0\nmax_high: 3\nblock_providers:\n  - Ollama\n")
        policy = load_policy(policy_file)
        assert policy.max_critical == 0
        assert policy.max_high == 3
        assert policy.block_providers == ["Ollama"]

    def test_load_empty_policy(self, tmp_path):
        policy_file = tmp_path / "policy.yml"
        policy_file.write_text("")
        policy = load_policy(policy_file)
        assert policy.max_critical is None

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/path.yml")

    def test_load_invalid_yaml(self, tmp_path):
        policy_file = tmp_path / "policy.yml"
        policy_file.write_text("{ invalid yaml [[[")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_policy(policy_file)


class TestEvaluatePolicy:
    def test_empty_result_passes(self):
        result = _make_result()
        policy = Policy(max_critical=0)
        passed, violations = evaluate_policy(result, policy)
        assert passed
        assert violations == []

    def test_max_critical_pass(self):
        result = _make_result(
            [
                _make_component(severity=Severity.high, score=60),
            ]
        )
        policy = Policy(max_critical=0)
        passed, _violations = evaluate_policy(result, policy)
        assert passed

    def test_max_critical_fail(self):
        result = _make_result(
            [
                _make_component(name="risky", severity=Severity.critical, score=80),
            ]
        )
        policy = Policy(max_critical=0)
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert len(violations) == 1
        assert "critical" in violations[0].lower()

    def test_max_high_fail(self):
        result = _make_result(
            [
                _make_component(name="h1", severity=Severity.high, score=60),
                _make_component(name="h2", severity=Severity.high, score=55),
                _make_component(name="h3", severity=Severity.high, score=65),
            ]
        )
        policy = Policy(max_high=2)
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert "3" in violations[0]

    def test_max_risk_score_fail(self):
        result = _make_result(
            [
                _make_component(name="danger", score=90, severity=Severity.critical),
            ]
        )
        policy = Policy(max_risk_score=75)
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert "90" in violations[0]

    def test_max_risk_score_pass(self):
        result = _make_result(
            [
                _make_component(score=50, severity=Severity.medium),
            ]
        )
        policy = Policy(max_risk_score=75)
        passed, _violations = evaluate_policy(result, policy)
        assert passed

    def test_block_providers_fail(self):
        result = _make_result(
            [
                _make_component(provider="Ollama"),
            ]
        )
        policy = Policy(block_providers=["Ollama"])
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert "Ollama" in violations[0]

    def test_block_providers_case_insensitive(self):
        result = _make_result(
            [
                _make_component(provider="OLLAMA"),
            ]
        )
        policy = Policy(block_providers=["ollama"])
        passed, _violations = evaluate_policy(result, policy)
        assert not passed

    def test_block_providers_pass(self):
        result = _make_result(
            [
                _make_component(provider="OpenAI"),
            ]
        )
        policy = Policy(block_providers=["Ollama"])
        passed, _violations = evaluate_policy(result, policy)
        assert passed

    def test_block_flags_fail(self):
        result = _make_result(
            [
                _make_component(name="leaked", flags=["hardcoded_api_key"]),
            ]
        )
        policy = Policy(block_flags=["hardcoded_api_key"])
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert "hardcoded_api_key" in violations[0]

    def test_block_flags_pass(self):
        result = _make_result(
            [
                _make_component(flags=["unpinned_model"]),
            ]
        )
        policy = Policy(block_flags=["hardcoded_api_key"])
        passed, _violations = evaluate_policy(result, policy)
        assert passed

    def test_multiple_violations(self):
        result = _make_result(
            [
                _make_component(
                    name="bad",
                    provider="Ollama",
                    severity=Severity.critical,
                    score=90,
                    flags=["hardcoded_api_key"],
                ),
            ]
        )
        policy = Policy(
            max_critical=0,
            max_risk_score=50,
            block_providers=["Ollama"],
            block_flags=["hardcoded_api_key"],
        )
        passed, violations = evaluate_policy(result, policy)
        assert not passed
        assert len(violations) >= 3  # critical count, risk score, provider, flag

    def test_default_policy_always_passes(self):
        result = _make_result(
            [
                _make_component(
                    severity=Severity.critical,
                    score=100,
                    flags=["hardcoded_api_key"],
                ),
            ]
        )
        policy = Policy()
        passed, violations = evaluate_policy(result, policy)
        assert passed
        assert violations == []
