"""Tests for StandaloneInterceptor and Cedar policy evaluation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from trusera_sdk import (
    CedarEvaluator,
    PolicyAction,
    PolicyDecision,
    RequestBlockedError,
    StandaloneInterceptor,
)
from trusera_sdk.cedar import parse_policy


# Helper to create mock responses
def create_mock_response(status_code=200, text="OK"):
    """Create a mock httpx.Response."""
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    response.content = text.encode()
    return response


# Cedar Policy Tests


def test_parse_simple_forbid_rule():
    """Test parsing a simple forbid rule."""
    policy = """
    forbid (
        principal,
        action == Action::"http",
        resource
    ) when {
        request.hostname == "deepseek.com"
    };
    """

    rules = parse_policy(policy)
    assert len(rules) == 1
    assert rules[0].action == PolicyAction.FORBID
    assert rules[0].field == "hostname"
    assert rules[0].operator == "=="
    assert rules[0].value == "deepseek.com"


def test_parse_multiple_rules():
    """Test parsing multiple rules."""
    policy = """
    // Block DeepSeek API
    forbid (principal, action == Action::"http", resource)
    when { request.hostname contains "deepseek.com" };

    // Allow OpenAI
    permit (principal, action == Action::"http", resource)
    when { request.hostname == "api.openai.com" };

    // Block POST to upload endpoints
    forbid (principal, action == Action::"http", resource)
    when { request.path contains "/upload" };
    """

    rules = parse_policy(policy)
    assert len(rules) == 3
    assert rules[0].action == PolicyAction.FORBID
    assert rules[0].operator == "contains"
    assert rules[1].action == PolicyAction.PERMIT
    assert rules[2].field == "path"


def test_parse_rule_with_comments():
    """Test that comments are properly stripped."""
    policy = """
    // This is a comment
    forbid (principal, action == Action::"http", resource)
    when { request.hostname == "bad.com" }; // inline comment
    """

    rules = parse_policy(policy)
    assert len(rules) == 1
    assert rules[0].value == "bad.com"


def test_evaluate_forbid_hostname():
    """Test forbid rule on hostname."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.hostname == "deepseek.com" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # Should deny deepseek.com
    result = evaluator.evaluate("https://deepseek.com/api/chat", "POST")
    assert result.decision == PolicyDecision.DENY
    assert "deepseek.com" in result.reason.lower()

    # Should allow others
    result = evaluator.evaluate("https://openai.com/api/chat", "POST")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_forbid_url_contains():
    """Test forbid rule with contains operator."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.url contains "api.deepseek.com" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # Should deny URLs containing the pattern
    result = evaluator.evaluate("https://api.deepseek.com/v1/chat", "POST")
    assert result.decision == PolicyDecision.DENY

    # Should allow others
    result = evaluator.evaluate("https://api.openai.com/v1/chat", "POST")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_forbid_method():
    """Test forbid rule on HTTP method."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.method == "DELETE" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # Should deny DELETE
    result = evaluator.evaluate("https://api.example.com/resource", "DELETE")
    assert result.decision == PolicyDecision.DENY

    # Should allow GET
    result = evaluator.evaluate("https://api.example.com/resource", "GET")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_forbid_path():
    """Test forbid rule on URL path."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.path contains "/admin" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # Should deny paths with /admin
    result = evaluator.evaluate("https://example.com/admin/users", "GET")
    assert result.decision == PolicyDecision.DENY

    # Should allow other paths
    result = evaluator.evaluate("https://example.com/api/users", "GET")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_permit_rule():
    """Test explicit permit rule."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.hostname contains "external" };

    permit (principal, action == Action::"http", resource)
    when { request.hostname == "external-trusted.com" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # external-trusted.com should be explicitly permitted (overrides forbid)
    # Note: In our implementation, forbid is checked first, so this will be denied
    # This tests that permit rules are evaluated
    result = evaluator.evaluate("https://external-trusted.com/api", "GET")
    # With forbid-first logic, this will be denied
    assert result.decision == PolicyDecision.DENY

    # Other external hosts should be denied
    result = evaluator.evaluate("https://external-bad.com/api", "GET")
    assert result.decision == PolicyDecision.DENY

    # Non-external hosts should be allowed (no rule matches)
    result = evaluator.evaluate("https://internal.com/api", "GET")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_no_rules():
    """Test evaluation with no rules (should allow all)."""
    evaluator = CedarEvaluator.from_text("")

    result = evaluator.evaluate("https://any.com/api", "POST")
    assert result.decision == PolicyDecision.ALLOW
    assert "no policy" in result.reason.lower() or "default" in result.reason.lower()


def test_evaluate_case_insensitive():
    """Test that evaluation is case-insensitive."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.hostname == "DEEPSEEK.COM" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    # Should deny regardless of case
    result = evaluator.evaluate("https://DeepSeek.com/api", "GET")
    assert result.decision == PolicyDecision.DENY


def test_evaluate_startswith_operator():
    """Test startswith operator."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.path startswith "/api/v1" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    result = evaluator.evaluate("https://example.com/api/v1/users", "GET")
    assert result.decision == PolicyDecision.DENY

    result = evaluator.evaluate("https://example.com/api/v2/users", "GET")
    assert result.decision == PolicyDecision.ALLOW


def test_evaluate_endswith_operator():
    """Test endswith operator."""
    policy = """
    forbid (principal, action == Action::"http", resource)
    when { request.path endswith ".exe" };
    """

    evaluator = CedarEvaluator.from_text(policy)

    result = evaluator.evaluate("https://example.com/download/malware.exe", "GET")
    assert result.decision == PolicyDecision.DENY

    result = evaluator.evaluate("https://example.com/download/file.pdf", "GET")
    assert result.decision == PolicyDecision.ALLOW


# StandaloneInterceptor Tests


def test_interceptor_initialization():
    """Test StandaloneInterceptor initialization."""
    interceptor = StandaloneInterceptor(
        enforcement="log",
        log_file="test.jsonl",
    )

    assert interceptor.enforcement == "log"
    assert interceptor.log_file == Path("test.jsonl")
    assert interceptor._installed is False


def test_interceptor_invalid_enforcement():
    """Test that invalid enforcement mode raises error."""
    with pytest.raises(ValueError, match="Invalid enforcement mode"):
        StandaloneInterceptor(enforcement="invalid")


def test_interceptor_missing_policy_file():
    """Test that missing policy file raises error."""
    with pytest.raises(FileNotFoundError):
        StandaloneInterceptor(policy_file="/nonexistent/policy.cedar")


def test_interceptor_install_uninstall():
    """Test install and uninstall of interceptor."""
    interceptor = StandaloneInterceptor(enforcement="log")

    # Should not be installed initially
    assert interceptor._installed is False

    # Install
    interceptor.install()
    assert interceptor._installed is True

    # Uninstall
    interceptor.uninstall()
    assert interceptor._installed is False


def test_interceptor_install_twice_raises():
    """Test that installing twice raises error."""
    interceptor = StandaloneInterceptor(enforcement="log")
    interceptor.install()

    with pytest.raises(RuntimeError, match="already installed"):
        interceptor.install()

    interceptor.uninstall()


def test_interceptor_uninstall_not_installed_raises():
    """Test that uninstalling when not installed raises error."""
    interceptor = StandaloneInterceptor(enforcement="log")

    with pytest.raises(RuntimeError, match="not installed"):
        interceptor.uninstall()


def test_interceptor_context_manager():
    """Test using interceptor as context manager."""
    with StandaloneInterceptor(enforcement="log") as interceptor:
        assert interceptor._installed is True

    # Should be uninstalled after context
    assert interceptor._installed is False


def test_interceptor_evaluate_and_enforce_no_policy():
    """Test evaluation without policy allows all requests."""
    interceptor = StandaloneInterceptor(enforcement="block")

    should_allow, reason = interceptor._evaluate_and_enforce("GET", "https://any.com/api")

    assert should_allow is True
    assert "no policy" in reason.lower()


def test_interceptor_evaluate_and_enforce_with_policy():
    """Test evaluation with policy."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "blocked.com" };
        """)
        policy_file = f.name

    try:
        interceptor = StandaloneInterceptor(policy_file=policy_file, enforcement="block")

        # Should block blocked.com
        should_allow, reason = interceptor._evaluate_and_enforce("GET", "https://blocked.com/api")
        assert should_allow is False
        assert "blocked.com" in reason.lower() or "forbidden" in reason.lower()

        # Should allow others
        should_allow, reason = interceptor._evaluate_and_enforce("GET", "https://allowed.com/api")
        assert should_allow is True

    finally:
        Path(policy_file).unlink(missing_ok=True)


def test_interceptor_block_mode_raises():
    """Test that block mode raises RequestBlockedError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "blocked.com" };
        """)
        policy_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="block",
            log_file=log_file,
        )

        # Create a mock request
        client = httpx.Client()
        request = client.build_request("GET", "https://blocked.com/api")

        # Mock original send
        mock_send = Mock(return_value=create_mock_response())

        with pytest.raises(RequestBlockedError, match="blocked by policy"):
            interceptor._intercept_sync_request(mock_send, request)

        # Check log file
        with open(log_file) as f:
            events = [json.loads(line) for line in f]
            assert len(events) == 1
            assert events[0]["enforcement_action"] == "blocked"
            assert events[0]["status_code"] is None  # Request never made

    finally:
        Path(policy_file).unlink(missing_ok=True)
        Path(log_file).unlink(missing_ok=True)


def test_interceptor_log_mode_allows_blocked():
    """Test that log mode allows requests even when blocked by policy."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "blocked.com" };
        """)
        policy_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="log",
            log_file=log_file,
        )

        # Create a mock request
        client = httpx.Client()
        request = client.build_request("GET", "https://blocked.com/api")

        # Mock original send
        mock_send = Mock(return_value=create_mock_response(200))

        # Should not raise in log mode
        response = interceptor._intercept_sync_request(mock_send, request)
        assert response.status_code == 200

        # Check log file
        with open(log_file) as f:
            events = [json.loads(line) for line in f]
            assert len(events) == 1
            assert events[0]["policy_decision"] == "deny"
            assert events[0]["enforcement_action"] == "warned"

    finally:
        Path(policy_file).unlink(missing_ok=True)
        Path(log_file).unlink(missing_ok=True)


def test_interceptor_warn_mode_prints_warning(capsys):
    """Test that warn mode prints warnings but allows requests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "blocked.com" };
        """)
        policy_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="warn",
        )

        # Create a mock request
        client = httpx.Client()
        request = client.build_request("GET", "https://blocked.com/api")

        # Mock original send
        mock_send = Mock(return_value=create_mock_response(200))

        # Should not raise in warn mode
        response = interceptor._intercept_sync_request(mock_send, request)
        assert response.status_code == 200

        # Check that warning was printed
        captured = capsys.readouterr()
        assert (
            "Policy violation" in captured.err or "WARNING" in captured.err or "⚠️" in captured.err
        )

    finally:
        Path(policy_file).unlink(missing_ok=True)


def test_interceptor_exclude_patterns():
    """Test that exclude patterns skip interception."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname contains "." };
        """)
        policy_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="block",
            log_file=log_file,
            exclude_patterns=[r"api\.trusera\."],
        )

        # Create a mock request to excluded URL
        client = httpx.Client()
        request = client.build_request("GET", "https://api.trusera.dev/test")

        # Mock original send
        mock_send = Mock(return_value=create_mock_response(200))

        # Should not raise (excluded from interception)
        response = interceptor._intercept_sync_request(mock_send, request)
        assert response.status_code == 200

        # Log file should be empty (excluded requests not logged)
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 0

    finally:
        Path(policy_file).unlink(missing_ok=True)
        Path(log_file).unlink(missing_ok=True)


def test_interceptor_jsonl_logging():
    """Test JSONL event logging format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            enforcement="log",
            log_file=log_file,
        )

        client = httpx.Client()

        # Log GET request
        request1 = client.build_request("GET", "https://example.com/api")
        mock_send1 = Mock(return_value=create_mock_response(200))
        interceptor._intercept_sync_request(mock_send1, request1)

        # Log POST request
        request2 = client.build_request("POST", "https://example.com/other")
        mock_send2 = Mock(return_value=create_mock_response(404))
        interceptor._intercept_sync_request(mock_send2, request2)

        # Verify JSONL format
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Parse each line as JSON
            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])

            # Check required fields
            assert "timestamp" in event1
            assert event1["method"] == "GET"
            assert event1["url"] == "https://example.com/api"
            assert event1["status_code"] == 200
            assert "duration_ms" in event1
            assert event1["policy_decision"] == "allow"
            assert event1["enforcement_action"] == "allowed"

            assert event2["method"] == "POST"
            assert event2["status_code"] == 404

    finally:
        Path(log_file).unlink(missing_ok=True)


def test_interceptor_get_stats():
    """Test get_stats method."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write(
            'forbid (principal, action == Action::"http", resource) when { request.hostname == "test.com" };'
        )
        policy_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="block",
            log_file="test.jsonl",
        )

        stats = interceptor.get_stats()
        assert stats["enforcement"] == "block"
        assert stats["policy_loaded"] is True
        assert stats["installed"] is False
        assert stats["events_logged"] == 0

    finally:
        Path(policy_file).unlink(missing_ok=True)


def test_interceptor_repr():
    """Test string representation."""
    interceptor = StandaloneInterceptor(enforcement="warn")
    repr_str = repr(interceptor)

    assert "StandaloneInterceptor" in repr_str
    assert "enforcement=warn" in repr_str
    assert "events=0" in repr_str


def test_interceptor_async_request_sync():
    """Test that async interceptor logic works (tested synchronously)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            enforcement="log",
            log_file=log_file,
        )

        # Test the logic without actually using async/await
        # The _evaluate_and_enforce method is the same for sync/async
        should_allow, reason = interceptor._evaluate_and_enforce(
            "GET", "https://async.example.com/api"
        )

        assert should_allow is True

    finally:
        Path(log_file).unlink(missing_ok=True)


def test_interceptor_async_block_mode_sync():
    """Test async block mode logic (tested synchronously)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "blocked.com" };
        """)
        policy_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            policy_file=policy_file,
            enforcement="block",
        )

        # Test the evaluation logic
        should_allow, reason = interceptor._evaluate_and_enforce("GET", "https://blocked.com/api")

        assert should_allow is False
        assert "blocked" in reason.lower() or "forbidden" in reason.lower()

    finally:
        Path(policy_file).unlink(missing_ok=True)


def test_cedar_evaluator_from_file():
    """Test loading Cedar policy from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cedar", delete=False) as f:
        f.write("""
        forbid (principal, action == Action::"http", resource)
        when { request.hostname == "test.com" };
        """)
        policy_file = f.name

    try:
        evaluator = CedarEvaluator.from_file(policy_file)
        result = evaluator.evaluate("https://test.com/api", "GET")
        assert result.decision == PolicyDecision.DENY

    finally:
        Path(policy_file).unlink(missing_ok=True)


def test_interceptor_should_intercept():
    """Test URL exclusion logic."""
    interceptor = StandaloneInterceptor(
        enforcement="log",
        exclude_patterns=[r"api\.trusera\.", r"localhost:\d+"],
    )

    # Should intercept normal URLs
    assert interceptor._should_intercept("https://example.com/api") is True

    # Should skip excluded patterns
    assert interceptor._should_intercept("https://api.trusera.dev/test") is False
    assert interceptor._should_intercept("http://localhost:8080/api") is False


def test_interceptor_log_file_creation():
    """Test that log file and parent directories are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "nested" / "dir" / "events.jsonl"

        interceptor = StandaloneInterceptor(
            enforcement="log",
            log_file=str(log_file),
        )

        # Log an event
        client = httpx.Client()
        request = client.build_request("GET", "https://example.com/api")
        mock_send = Mock(return_value=create_mock_response(200))
        interceptor._intercept_sync_request(mock_send, request)

        # Verify file was created with parent directories
        assert log_file.exists()
        assert log_file.parent.exists()


def test_interceptor_error_logging():
    """Test that errors during request are logged."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_file = f.name

    try:
        interceptor = StandaloneInterceptor(
            enforcement="log",
            log_file=log_file,
        )

        client = httpx.Client()
        request = client.build_request("GET", "https://example.com/api")

        # Mock original send to raise an error
        def mock_send_error(req, **kwargs):
            raise httpx.ConnectError("Connection failed")

        with pytest.raises(httpx.ConnectError):
            interceptor._intercept_sync_request(mock_send_error, request)

        # Check that error was logged
        with open(log_file) as f:
            events = [json.loads(line) for line in f]
            assert len(events) == 1
            assert events[0]["enforcement_action"] == "error"
            assert "Connection failed" in events[0]["error"]

    finally:
        Path(log_file).unlink(missing_ok=True)
