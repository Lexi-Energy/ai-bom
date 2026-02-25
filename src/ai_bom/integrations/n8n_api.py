"""n8n REST API client for live workflow scanning."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAX_PAGES = 1000


class N8nAPIError(Exception):
    """Base exception for n8n API errors."""


class N8nAuthError(N8nAPIError):
    """Raised when authentication fails (401/403)."""


class N8nConnectionError(N8nAPIError):
    """Raised when the n8n instance cannot be reached."""


class N8nAPIClient:
    """Client for the n8n REST API (v1).

    Args:
        base_url: Base URL of the n8n instance (e.g. ``http://localhost:5678``).
        api_key: n8n API key for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-N8N-API-KEY": api_key})

    def _request(self, method: str, path: str) -> Any:
        """Make an authenticated request to the n8n API.

        Args:
            method: HTTP method.
            path: API path (e.g. ``/api/v1/workflows``).

        Returns:
            Parsed JSON response body.

        Raises:
            N8nAuthError: On 401/403 responses.
            N8nConnectionError: On network-level failures.
            N8nAPIError: On any other non-2xx response.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise N8nConnectionError(f"Could not connect to n8n at {self.base_url}: {exc}") from exc
        except requests.Timeout as exc:
            raise N8nConnectionError(
                f"Request to n8n timed out after {self.timeout}s: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise N8nAPIError(f"Request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise N8nAuthError(
                f"Authentication failed (HTTP {resp.status_code}). Check your API key."
            )

        if not resp.ok:
            raise N8nAPIError(f"n8n API returned HTTP {resp.status_code}: {resp.text}")

        return resp.json()

    def list_workflows(self) -> list[dict[str, Any]]:
        """Fetch all workflows from the n8n instance.

        Handles cursor-based pagination automatically.

        Returns:
            List of workflow data dicts (same structure as exported JSON).
        """
        workflows: list[dict[str, Any]] = []
        cursor: str | None = None

        for _page in range(MAX_PAGES):
            path = "/api/v1/workflows"
            if cursor:
                path += f"?cursor={urllib.parse.quote(cursor, safe='')}"

            data = self._request("GET", path)

            workflows.extend(data.get("data", []))

            next_cursor = data.get("nextCursor")
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            logger.warning(
                "n8n pagination hit safety limit of %d pages; results may be incomplete",
                MAX_PAGES,
            )

        return workflows

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """Fetch a single workflow by ID.

        Args:
            workflow_id: The n8n workflow ID.

        Returns:
            Workflow data dict.
        """
        result: dict[str, Any] = self._request("GET", f"/api/v1/workflows/{workflow_id}")
        return result
