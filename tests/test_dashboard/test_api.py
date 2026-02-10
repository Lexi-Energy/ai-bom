"""Tests for the dashboard REST API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture()
def _db_path(tmp_path: Path):
    """Override the DB_PATH so tests use a temp database."""
    db_path = tmp_path / "test_api.db"
    with patch("ai_bom.dashboard.db.DB_PATH", db_path):
        from ai_bom.dashboard.db import init_db

        init_db(db_path=db_path)
        yield db_path


@pytest.fixture()
def client(_db_path: Path):
    """Create a TestClient for the dashboard app."""
    from ai_bom.dashboard import create_app

    app = create_app()
    return TestClient(app)


def _scan_payload(scan_id: str = "test-scan-001", target: str = "/home/user/project") -> dict:
    return {
        "id": scan_id,
        "timestamp": "2025-01-15T10:00:00Z",
        "target_path": target,
        "summary": {
            "total_components": 2,
            "by_severity": {"high": 1, "low": 1},
            "highest_risk_score": 75,
        },
        "components": [
            {
                "name": "openai",
                "type": "llm_provider",
                "provider": "openai",
                "risk": {"score": 75, "severity": "high"},
                "location": {"file_path": "app.py"},
            },
            {
                "name": "gpt-4",
                "type": "model",
                "provider": "openai",
                "risk": {"score": 30, "severity": "low"},
                "location": {"file_path": "app.py"},
            },
        ],
        "scan_duration": 1.5,
        "ai_bom_version": "0.1.0",
    }


class TestDashboardRoot:
    def test_serves_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AI-BOM Dashboard" in resp.text


class TestListScans:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get("/api/scans")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_after_create(self, client: TestClient) -> None:
        client.post("/api/scans", json=_scan_payload())
        resp = client.get("/api/scans")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "test-scan-001"
        assert "components" not in data[0]  # listing omits full components


class TestCreateScan:
    def test_success(self, client: TestClient) -> None:
        resp = client.post("/api/scans", json=_scan_payload())
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "test-scan-001"
        assert body["status"] == "saved"

    def test_missing_required_fields(self, client: TestClient) -> None:
        resp = client.post("/api/scans", json={"id": "bad"})
        assert resp.status_code == 422

    def test_auto_generates_id(self, client: TestClient) -> None:
        payload = _scan_payload()
        del payload["id"]
        resp = client.post("/api/scans", json=payload)
        assert resp.status_code == 201
        assert resp.json()["id"]  # should have a UUID


class TestGetScan:
    def test_found(self, client: TestClient) -> None:
        client.post("/api/scans", json=_scan_payload())
        resp = client.get("/api/scans/test-scan-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-scan-001"
        assert len(data["components"]) == 2

    def test_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/scans/nonexistent")
        assert resp.status_code == 404


class TestDeleteScan:
    def test_delete_existing(self, client: TestClient) -> None:
        client.post("/api/scans", json=_scan_payload())
        resp = client.delete("/api/scans/test-scan-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        # Verify it's gone
        resp = client.get("/api/scans/test-scan-001")
        assert resp.status_code == 404

    def test_delete_not_found(self, client: TestClient) -> None:
        resp = client.delete("/api/scans/nope")
        assert resp.status_code == 404


class TestCompareScans:
    def test_compare_two_scans(self, client: TestClient) -> None:
        client.post("/api/scans", json=_scan_payload("scan-a", "/path/a"))
        # Second scan has different components
        payload_b = _scan_payload("scan-b", "/path/b")
        payload_b["components"] = [
            {
                "name": "openai",
                "type": "llm_provider",
                "provider": "openai",
                "risk": {"score": 75, "severity": "high"},
                "location": {"file_path": "app.py"},
            },
            {
                "name": "anthropic",
                "type": "llm_provider",
                "provider": "anthropic",
                "risk": {"score": 60, "severity": "medium"},
                "location": {"file_path": "bot.py"},
            },
        ]
        client.post("/api/scans", json=payload_b)

        resp = client.get("/api/compare", params={"scan_id_1": "scan-a", "scan_id_2": "scan-b"})
        assert resp.status_code == 200
        data = resp.json()
        assert "anthropic" in data["added"]
        assert "gpt-4" in data["removed"]
        assert "openai" in data["common"]

    def test_compare_missing_scan(self, client: TestClient) -> None:
        client.post("/api/scans", json=_scan_payload("scan-x"))
        resp = client.get("/api/compare", params={"scan_id_1": "scan-x", "scan_id_2": "missing"})
        assert resp.status_code == 404
