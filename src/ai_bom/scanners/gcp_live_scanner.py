"""GCP Live Scanner — discovers AI/ML services in a live GCP project."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ai_bom.models import AIComponent, ComponentType, SourceLocation, UsageType
from ai_bom.scanners.base import BaseScanner

logger = logging.getLogger(__name__)

try:
    from google.api_core.exceptions import GoogleAPICallError, PermissionDenied
    from google.cloud import aiplatform

    _HAS_GCP = True
except ImportError:
    aiplatform = None  # type: ignore[assignment]
    _HAS_GCP = False

    class GoogleAPICallError(Exception):  # type: ignore[no-redef]
        pass

    class PermissionDenied(Exception):  # type: ignore[no-redef]  # noqa: N818
        pass


class GCPLiveScanner(BaseScanner):
    """Scan GCP project for managed AI/ML services via Google Cloud SDK."""

    name = "gcp-live"
    description = "Scan GCP project for AI/ML services"
    enabled: bool = False

    def supports(self, path: Path) -> bool:
        return self.enabled and _HAS_GCP

    def scan(self, path: Path) -> list[AIComponent]:
        if not _HAS_GCP:
            logger.warning("google-cloud-aiplatform is not installed — skipping GCP live scan")
            return []

        components: list[AIComponent] = []
        try:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
            if not project:
                logger.error("No GCP project configured. Set GOOGLE_CLOUD_PROJECT env var.")
                return []
            aiplatform.init(project=project)
            components.extend(self._scan_vertex_models())
            components.extend(self._scan_vertex_endpoints())
            components.extend(self._scan_vertex_custom_jobs())
        except (GoogleAPICallError, PermissionDenied) as exc:
            logger.error("GCP API error: %s", exc)
        except Exception as exc:
            logger.error("GCP scan error: %s", exc)
        return components

    # ------------------------------------------------------------------
    # Vertex AI Models
    # ------------------------------------------------------------------
    def _scan_vertex_models(self) -> list[AIComponent]:
        components: list[AIComponent] = []
        try:
            models = aiplatform.Model.list()
            for model in models:
                model_name = model.display_name
                model_id = model.resource_name
                components.append(
                    AIComponent(
                        name=model_name,
                        type=ComponentType.model,
                        provider="GCP Vertex AI",
                        model_name=model_name,
                        location=SourceLocation(
                            file_path=f"gcp://vertex-ai/model/{model_id}",
                        ),
                        usage_type=UsageType.completion,
                        source="cloud-live",
                    )
                )
        except Exception as exc:
            logger.debug("Vertex AI list_models failed: %s", exc)
        return components

    # ------------------------------------------------------------------
    # Vertex AI Endpoints
    # ------------------------------------------------------------------
    def _scan_vertex_endpoints(self) -> list[AIComponent]:
        components: list[AIComponent] = []
        try:
            endpoints = aiplatform.Endpoint.list()
            for ep in endpoints:
                ep_name = ep.display_name
                ep_id = ep.resource_name
                components.append(
                    AIComponent(
                        name=ep_name,
                        type=ComponentType.endpoint,
                        provider="GCP Vertex AI",
                        location=SourceLocation(
                            file_path=f"gcp://vertex-ai/endpoint/{ep_id}",
                        ),
                        usage_type=UsageType.completion,
                        source="cloud-live",
                    )
                )
        except Exception as exc:
            logger.debug("Vertex AI list_endpoints failed: %s", exc)
        return components

    # ------------------------------------------------------------------
    # Vertex AI Custom Jobs
    # ------------------------------------------------------------------
    def _scan_vertex_custom_jobs(self) -> list[AIComponent]:
        components: list[AIComponent] = []
        try:
            jobs = aiplatform.CustomJob.list()
            for job in jobs:
                job_name = job.display_name
                job_id = job.resource_name
                components.append(
                    AIComponent(
                        name=job_name,
                        type=ComponentType.model,
                        provider="GCP Vertex AI",
                        location=SourceLocation(
                            file_path=f"gcp://vertex-ai/custom-job/{job_id}",
                        ),
                        usage_type=UsageType.unknown,
                        source="cloud-live",
                    )
                )
        except Exception as exc:
            logger.debug("Vertex AI list_custom_jobs failed: %s", exc)
        return components
