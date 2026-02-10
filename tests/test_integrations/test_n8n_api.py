"""Tests for the n8n API client and live scanner integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from ai_bom.integrations.n8n_api import (
    N8nAPIClient,
    N8nAPIError,
    N8nAuthError,
    N8nConnectionError,
)
from ai_bom.scanners.n8n_scanner import N8nScanner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> N8nAPIClient:
    return N8nAPIClient("http://localhost:5678", "test-api-key")


SAMPLE_WORKFLOW = {
    "id": "wf1",
    "name": "AI Chat Bot",
    "nodes": [
        {
            "name": "Agent",
            "type": "@n8n/n8n-nodes-langchain.agent",
            "parameters": {},
        },
        {
            "name": "OpenAI Chat",
            "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
            "parameters": {"model": "gpt-4"},
        },
    ],
    "connections": {"Agent": {"main": [[{"node": "OpenAI Chat", "type": "main", "index": 0}]]}},
}


def _mock_response(json_data, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# ---------------------------------------------------------------------------
# N8nAPIClient unit tests
# ---------------------------------------------------------------------------


class TestN8nAPIClientInit:
    def test_strips_trailing_slash(self):
        c = N8nAPIClient("http://host:5678/", "key")
        assert c.base_url == "http://host:5678"

    def test_sets_auth_header(self):
        c = N8nAPIClient("http://host:5678", "my-key")
        assert c.session.headers["X-N8N-API-KEY"] == "my-key"


class TestListWorkflows:
    def test_single_page(self, client: N8nAPIClient):
        payload = {"data": [SAMPLE_WORKFLOW], "nextCursor": None}
        with patch.object(client.session, "request", return_value=_mock_response(payload)):
            workflows = client.list_workflows()
        assert len(workflows) == 1
        assert workflows[0]["id"] == "wf1"

    def test_pagination(self, client: N8nAPIClient):
        page1 = {"data": [SAMPLE_WORKFLOW], "nextCursor": "cursor-abc"}
        page2_wf = {**SAMPLE_WORKFLOW, "id": "wf2", "name": "Second"}
        page2 = {"data": [page2_wf], "nextCursor": None}

        call_count = 0

        def _side_effect(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(page1)
            return _mock_response(page2)

        with patch.object(client.session, "request", side_effect=_side_effect):
            workflows = client.list_workflows()

        assert len(workflows) == 2
        assert workflows[0]["id"] == "wf1"
        assert workflows[1]["id"] == "wf2"

    def test_empty_result(self, client: N8nAPIClient):
        payload = {"data": [], "nextCursor": None}
        with patch.object(client.session, "request", return_value=_mock_response(payload)):
            workflows = client.list_workflows()
        assert workflows == []


class TestGetWorkflow:
    def test_returns_workflow(self, client: N8nAPIClient):
        with patch.object(client.session, "request", return_value=_mock_response(SAMPLE_WORKFLOW)):
            wf = client.get_workflow("wf1")
        assert wf["name"] == "AI Chat Bot"


class TestErrorHandling:
    def test_auth_error_401(self, client: N8nAPIClient):
        with (
            patch.object(client.session, "request", return_value=_mock_response({}, 401)),
            pytest.raises(N8nAuthError, match="Authentication failed"),
        ):
            client.list_workflows()

    def test_auth_error_403(self, client: N8nAPIClient):
        with (
            patch.object(client.session, "request", return_value=_mock_response({}, 403)),
            pytest.raises(N8nAuthError, match="Authentication failed"),
        ):
            client.list_workflows()

    def test_server_error_500(self, client: N8nAPIClient):
        with (
            patch.object(client.session, "request", return_value=_mock_response({}, 500)),
            pytest.raises(N8nAPIError, match="HTTP 500"),
        ):
            client.list_workflows()

    def test_connection_error(self, client: N8nAPIClient):
        with (
            patch.object(
                client.session,
                "request",
                side_effect=requests.ConnectionError("refused"),
            ),
            pytest.raises(N8nConnectionError, match="Could not connect"),
        ):
            client.list_workflows()

    def test_timeout_error(self, client: N8nAPIClient):
        with (
            patch.object(
                client.session,
                "request",
                side_effect=requests.Timeout("timed out"),
            ),
            pytest.raises(N8nConnectionError, match="timed out"),
        ):
            client.list_workflows()

    def test_generic_request_error(self, client: N8nAPIClient):
        with (
            patch.object(
                client.session,
                "request",
                side_effect=requests.RequestException("something broke"),
            ),
            pytest.raises(N8nAPIError, match="Request failed"),
        ):
            client.list_workflows()


# ---------------------------------------------------------------------------
# scan_from_api integration (with mocked API)
# ---------------------------------------------------------------------------


class TestScanFromApi:
    def test_extracts_components_from_api(self):
        """End-to-end: mock API returns workflow, scanner produces components."""
        client = MagicMock(spec=N8nAPIClient)
        client.list_workflows.return_value = [SAMPLE_WORKFLOW]

        scanner = N8nScanner()
        components = scanner.scan_from_api(client)

        # The sample workflow has an agent node and an OpenAI LLM node
        assert len(components) >= 2

        names = {c.name for c in components}
        assert "Agent" in names
        assert "OpenAI Chat" in names

        # Check that the OpenAI component has correct provider
        openai_comp = next(c for c in components if c.name == "OpenAI Chat")
        assert openai_comp.provider == "OpenAI"
        assert openai_comp.model_name == "gpt-4"

    def test_skips_invalid_workflows(self):
        """Workflows missing nodes/connections are silently skipped."""
        client = MagicMock(spec=N8nAPIClient)
        client.list_workflows.return_value = [
            {"id": "bad", "name": "Not a workflow"},  # missing nodes/connections
            SAMPLE_WORKFLOW,
        ]

        scanner = N8nScanner()
        components = scanner.scan_from_api(client)
        assert len(components) >= 2  # only from SAMPLE_WORKFLOW

    def test_empty_workflow_list(self):
        client = MagicMock(spec=N8nAPIClient)
        client.list_workflows.return_value = []

        scanner = N8nScanner()
        components = scanner.scan_from_api(client)
        assert components == []

    def test_synthetic_path_contains_workflow_id(self):
        client = MagicMock(spec=N8nAPIClient)
        client.list_workflows.return_value = [SAMPLE_WORKFLOW]

        scanner = N8nScanner()
        components = scanner.scan_from_api(client)

        for comp in components:
            assert "wf1" in comp.location.file_path

    def test_workflow_metadata_populated(self):
        client = MagicMock(spec=N8nAPIClient)
        client.list_workflows.return_value = [SAMPLE_WORKFLOW]

        scanner = N8nScanner()
        components = scanner.scan_from_api(client)

        for comp in components:
            assert comp.metadata["workflow_name"] == "AI Chat Bot"
            assert comp.metadata["workflow_id"] == "wf1"
