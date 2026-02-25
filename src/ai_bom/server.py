"""REST API server for AI-BOM scanning.

Provides HTTP endpoints for scanning directories and retrieving results.
Start with: ai-bom serve --port 8080

Security (SEC-01):
- AIBOM_SCAN_ROOT: restricts scan paths to a directory subtree (default: cwd)
- AIBOM_CORS_ORIGINS: comma-separated allow-list (default: http://localhost:8080)
- AIBOM_API_KEY: optional Bearer token for all endpoints except /health
- Error messages never echo user-supplied paths back to the caller
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ai_bom import __version__
from ai_bom.models import ScanResult
from ai_bom.scanners import get_all_scanners
from ai_bom.utils.risk_scorer import score_component


def create_server_app() -> Any:
    """Create and configure the FastAPI server application.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(
            "Server dependencies not installed. Install with: pip install ai-bom[server]"
        ) from e

    # --- SEC-01: security configuration from environment ---
    scan_root = Path(os.environ.get("AIBOM_SCAN_ROOT", ".")).resolve()
    cors_origins = os.environ.get("AIBOM_CORS_ORIGINS", "http://localhost:8080").split(",")
    api_key = os.environ.get("AIBOM_API_KEY", "")

    app = FastAPI(
        title="AI-BOM API",
        description="AI Bill of Materials — REST API for scanning AI/LLM components",
        version=__version__,
    )

    # --- SEC-01: optional Bearer-token auth middleware ---
    if api_key:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse as StarletteJSONResponse

        class APIKeyMiddleware(BaseHTTPMiddleware):
            """Reject requests without a valid Bearer token.

            The /health endpoint is exempt so load-balancers can probe
            without credentials.
            """

            async def dispatch(self, request, call_next):  # type: ignore[override]
                if request.url.path == "/health":
                    return await call_next(request)
                auth = request.headers.get("authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != api_key:
                    return StarletteJSONResponse(
                        {"error": "Unauthorized"}, status_code=401
                    )
                return await call_next(request)

        app.add_middleware(APIKeyMiddleware)

    # --- SEC-01: restricted CORS origins (no wildcard) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    class ScanRequest(BaseModel):
        path: str = "."
        deep: bool = False
        severity: str | None = None

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str = __version__

    class VersionResponse(BaseModel):
        version: str = __version__
        name: str = "ai-bom"

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    @app.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        """Version info endpoint."""
        return VersionResponse()

    @app.post("/scan")
    async def scan_endpoint(request: ScanRequest) -> dict:
        """Scan a directory path for AI/LLM components.

        Args:
            request: Scan request with path and options.

        Returns:
            Scan result as JSON dict.
        """
        # --- SEC-01: path traversal protection ---
        scan_path = Path(request.path).resolve()
        try:
            scan_path.relative_to(scan_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")

        if not scan_path.exists():
            # Generic message — never echo the user-supplied path
            raise HTTPException(status_code=404, detail="Invalid scan target")

        result = ScanResult(target_path=str(scan_path))
        start_time = time.time()

        scanners = get_all_scanners()
        if request.deep:
            from ai_bom.scanners.ast_scanner import ASTScanner

            for s in scanners:
                if isinstance(s, ASTScanner):
                    s.enabled = True

        for scanner in scanners:
            if not scanner.supports(scan_path):
                continue
            try:
                components = scanner.scan(scan_path)
                for comp in components:
                    comp.risk = score_component(comp)
                result.components.extend(components)
            except Exception:
                pass

        result.summary.scan_duration_seconds = time.time() - start_time
        result.build_summary()

        # Apply severity filter if specified
        if request.severity:
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            min_level = severity_order.get(request.severity.lower(), 0)
            result.components = [
                c
                for c in result.components
                if severity_order.get(c.risk.severity.value, 0) >= min_level
            ]
            result.build_summary()

        return result.model_dump(mode="json")

    @app.get("/scanners")
    async def list_scanners() -> list[dict]:
        """List all available scanners."""
        scanners = get_all_scanners()
        return [
            {
                "name": s.name,
                "description": s.description,
                "enabled": getattr(s, "enabled", True),
            }
            for s in scanners
        ]

    return app
