"""Shared fixtures for tests."""

from unittest.mock import Mock

import httpx
import pytest

try:
    from pytest_httpx import HTTPXMock

    HTTPX_MOCK_AVAILABLE = True
except ImportError:
    HTTPX_MOCK_AVAILABLE = False

from trusera_sdk import TruseraClient
from trusera_sdk.cedar import EvaluationResult, PolicyDecision


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock httpx.Client for testing."""
    mock_client = Mock(spec=httpx.Client)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "agent_123"}
    mock_response.raise_for_status = Mock()

    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response
    mock_client.close = Mock()

    # Patch httpx.Client constructor
    monkeypatch.setattr("httpx.Client", lambda **kwargs: mock_client)

    return mock_client


@pytest.fixture
def trusera_client(mock_httpx_client):
    """Create a TruseraClient instance with mocked HTTP."""
    client = TruseraClient(
        api_key="tsk_test_key",
        base_url="https://api.test.trusera.dev",
        flush_interval=0.1,  # Fast for testing
        batch_size=5,
    )
    client.set_agent_id("agent_test_123")
    yield client
    client.close()


@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "ok"}
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def httpx_mock():
    """
    Mock httpx requests using pytest-httpx if available.

    Falls back to a simple mock if pytest-httpx is not installed.
    """
    if HTTPX_MOCK_AVAILABLE:
        mock = HTTPXMock()
        yield mock
    else:
        # Simple fallback mock for when pytest-httpx is not installed
        class SimpleMock:
            def __init__(self):
                self.responses = []

            def add_response(self, url=None, status_code=200, text="", **kwargs):
                self.responses.append(
                    {"url": url, "status_code": status_code, "text": text, **kwargs}
                )

        yield SimpleMock()


# ---------------------------------------------------------------------------
# Interceptor-related fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_all_cache():
    """A mock PolicyCache that allows everything."""
    cache = Mock()
    cache.evaluate_request.return_value = EvaluationResult(
        decision=PolicyDecision.ALLOW,
        reason="Mock allow-all",
    )
    cache.evaluate_action.return_value = EvaluationResult(
        decision=PolicyDecision.ALLOW,
        reason="Mock allow-all",
    )
    return cache


@pytest.fixture
def deny_all_cache():
    """A mock PolicyCache that denies everything."""
    cache = Mock()
    cache.evaluate_request.return_value = EvaluationResult(
        decision=PolicyDecision.DENY,
        reason="Mock deny-all",
    )
    cache.evaluate_action.return_value = EvaluationResult(
        decision=PolicyDecision.DENY,
        reason="Mock deny-all",
    )
    return cache
