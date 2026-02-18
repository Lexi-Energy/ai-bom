"""Tests for PolicyViolationError."""

from trusera_sdk.exceptions import PolicyViolationError


def test_policy_violation_error_attributes():
    err = PolicyViolationError(
        action="http",
        target="GET https://evil.com",
        reason="Blocked by policy",
        policy_id="pol_123",
    )
    assert err.action == "http"
    assert err.target == "GET https://evil.com"
    assert err.reason == "Blocked by policy"
    assert err.policy_id == "pol_123"
    assert "http" in str(err)
    assert "evil.com" in str(err)


def test_policy_violation_error_no_policy_id():
    err = PolicyViolationError(action="tool_call", target="search", reason="Denied")
    assert err.policy_id is None
    assert "tool_call" in str(err)


def test_policy_violation_error_is_exception():
    assert issubclass(PolicyViolationError, Exception)
