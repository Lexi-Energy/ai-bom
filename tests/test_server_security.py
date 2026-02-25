"""Security tests for the AI-BOM REST API server.

SEC-01: Path traversal, CORS, API key auth, generic error messages.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def app_factory():
    """Import factory inside fixture to allow env var patching."""
    from ai_bom.server import create_server_app
    return create_server_app


@pytest.fixture
def client(app_factory):
    """Create a test client with default settings."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    app = app_factory()
    return TestClient(app)


class TestPathTraversal:
    """SEC-01: Verify path traversal protection."""

    def test_path_outside_scan_root_rejected(self, app_factory):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        with patch.dict(os.environ, {"AIBOM_SCAN_ROOT": "/tmp/safe"}):
            app = app_factory()
            client = TestClient(app)
            resp = client.post("/scan", json={"path": "/etc/passwd"})
            assert resp.status_code == 403

    def test_relative_traversal_rejected(self, app_factory):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        with patch.dict(os.environ, {"AIBOM_SCAN_ROOT": "/tmp/safe"}):
            app = app_factory()
            client = TestClient(app)
            resp = client.post("/scan", json={"path": "/tmp/safe/../../etc/passwd"})
            assert resp.status_code == 403


class TestCORS:
    """SEC-01: Verify CORS is not wildcard."""

    def test_default_cors_not_wildcard(self, client):
        resp = client.options(
            "/scan",
            headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "POST"},
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin != "*"

    def test_custom_cors_origins(self, app_factory):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        with patch.dict(os.environ, {"AIBOM_CORS_ORIGINS": "https://app.example.com"}):
            app = app_factory()
            client = TestClient(app)
            resp = client.options(
                "/scan",
                headers={
                    "Origin": "https://app.example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            allow_origin = resp.headers.get("access-control-allow-origin", "")
            assert allow_origin in ("https://app.example.com", "")


class TestAPIKeyAuth:
    """SEC-01: Verify optional API key middleware."""

    def test_no_auth_required_by_default(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_auth_required_when_key_set(self, app_factory):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        with patch.dict(os.environ, {"AIBOM_API_KEY": "test-secret"}):
            app = app_factory()
            client = TestClient(app)
            # Health is exempt
            assert client.get("/health").status_code == 200
            # Scan requires auth
            resp = client.post("/scan", json={"path": "."})
            assert resp.status_code == 401
            # With correct key
            resp = client.post(
                "/scan",
                json={"path": "."},
                headers={"Authorization": "Bearer test-secret"},
            )
            assert resp.status_code != 401


class TestErrorMessages:
    """SEC-01: Verify error messages don't leak paths."""

    def test_not_found_generic_message(self, client):
        resp = client.post("/scan", json={"path": "/nonexistent/secret/path"})
        body = resp.json()
        assert "/nonexistent/secret/path" not in str(body.get("detail", ""))
