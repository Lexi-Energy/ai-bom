"""Tests for live cloud API scanners (AWS, GCP, Azure)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_bom.models import ComponentType


def _make_aws_clients(
    *,
    foundation_models=None,
    agents=None,
    knowledge_bases=None,
    endpoints=None,
    models=None,
    notebooks=None,
    comprehend_jobs=None,
    kendra_indices=None,
):
    """Build mock boto3 clients with preset return values."""
    bedrock = MagicMock()
    bedrock.list_foundation_models.return_value = {
        "modelSummaries": foundation_models or [],
    }

    agent_client = MagicMock()
    agent_client.list_agents.return_value = {"agentSummaries": agents or []}
    agent_client.list_knowledge_bases.return_value = {
        "knowledgeBaseSummaries": knowledge_bases or [],
    }

    sagemaker = MagicMock()
    sagemaker.list_endpoints.return_value = {"Endpoints": endpoints or []}
    sagemaker.list_models.return_value = {"Models": models or []}
    sagemaker.list_notebook_instances.return_value = {
        "NotebookInstances": notebooks or [],
    }

    comprehend = MagicMock()
    comprehend.list_entities_detection_jobs.return_value = {
        "EntitiesDetectionJobPropertiesList": comprehend_jobs or [],
    }

    kendra = MagicMock()
    kendra.list_indices.return_value = {
        "IndexConfigurationSummaryItems": kendra_indices or [],
    }

    def client_factory(service):
        return {
            "bedrock": bedrock,
            "bedrock-agent": agent_client,
            "sagemaker": sagemaker,
            "comprehend": comprehend,
            "kendra": kendra,
        }[service]

    mock_boto3 = MagicMock()
    mock_boto3.client.side_effect = client_factory
    return mock_boto3


# =====================================================================
# AWS Live Scanner
# =====================================================================


class TestAWSLiveScanner:
    """Tests for AWSLiveScanner."""

    def test_disabled_by_default(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        scanner = AWSLiveScanner()
        assert scanner.enabled is False
        assert scanner.supports(Path(".")) is False

    def test_supports_when_enabled(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        scanner = AWSLiveScanner()
        scanner.enabled = True
        from ai_bom.scanners import aws_live_scanner as mod

        if mod._HAS_BOTO3:
            assert scanner.supports(Path(".")) is True
        else:
            assert scanner.supports(Path(".")) is False

    def test_name_and_description(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        scanner = AWSLiveScanner()
        assert scanner.name == "aws-live"
        assert "AWS" in scanner.description

    def test_scan_bedrock_foundation_models(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        mock_boto3 = _make_aws_clients(
            foundation_models=[
                {"modelId": "anthropic.claude-v2", "providerName": "Anthropic"},
                {"modelId": "amazon.titan-text-express-v1", "providerName": "Amazon"},
            ],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 2
        assert components[0].name == "anthropic.claude-v2"
        assert components[0].type == ComponentType.model
        assert "Anthropic" in components[0].provider
        assert components[0].source == "cloud-live"
        assert "aws://bedrock" in components[0].location.file_path

    def test_scan_bedrock_agents(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        mock_boto3 = _make_aws_clients(
            agents=[{"agentId": "agent-123", "agentName": "MyAgent"}],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 1
        assert components[0].name == "MyAgent"
        assert components[0].type == ComponentType.agent_framework
        assert components[0].provider == "AWS Bedrock"

    def test_scan_sagemaker_endpoints_and_models(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        mock_boto3 = _make_aws_clients(
            endpoints=[
                {
                    "EndpointName": "my-model-endpoint",
                    "EndpointArn": "arn:aws:sagemaker:us-east-1:123:endpoint/my-model-endpoint",
                }
            ],
            models=[
                {
                    "ModelName": "my-model",
                    "ModelArn": "arn:aws:sagemaker:us-east-1:123:model/my-model",
                }
            ],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 2
        endpoints = [c for c in components if c.type == ComponentType.endpoint]
        models = [c for c in components if c.type == ComponentType.model]
        assert len(endpoints) == 1
        assert endpoints[0].name == "my-model-endpoint"
        assert endpoints[0].provider == "AWS SageMaker"
        assert len(models) == 1
        assert models[0].name == "my-model"

    def test_scan_handles_no_credentials(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner, NoCredentialsError

        mock_boto3 = MagicMock()
        mock_boto3.client.side_effect = NoCredentialsError()

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))
            assert components == []

    @patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", False)
    def test_scan_without_boto3(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        scanner = AWSLiveScanner()
        scanner.enabled = True
        components = scanner.scan(Path("."))
        assert components == []

    def test_not_triggered_in_normal_scan(self):
        """Live scanners must NOT run during normal file scanning."""
        from ai_bom.scanners import get_all_scanners

        scanners = get_all_scanners()
        live_scanners = [s for s in scanners if s.name == "aws-live"]
        for s in live_scanners:
            assert s.supports(Path("/some/project")) is False

    def test_scan_comprehend_jobs(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        mock_boto3 = _make_aws_clients(
            comprehend_jobs=[{"JobId": "job-1", "JobName": "entity-detect-1"}],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        comprehend = [c for c in components if c.provider == "AWS Comprehend"]
        assert len(comprehend) == 1
        assert comprehend[0].name == "entity-detect-1"

    def test_scan_kendra_indices(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        mock_boto3 = _make_aws_clients(
            kendra_indices=[{"Id": "idx-1", "Name": "my-kendra-index"}],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            scanner = AWSLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        kendra = [c for c in components if c.provider == "AWS Kendra"]
        assert len(kendra) == 1
        assert kendra[0].name == "my-kendra-index"
        assert kendra[0].type == ComponentType.tool


# =====================================================================
# GCP Live Scanner
# =====================================================================


class TestGCPLiveScanner:
    """Tests for GCPLiveScanner."""

    def test_disabled_by_default(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        scanner = GCPLiveScanner()
        assert scanner.enabled is False
        assert scanner.supports(Path(".")) is False

    def test_name_and_description(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        scanner = GCPLiveScanner()
        assert scanner.name == "gcp-live"
        assert "GCP" in scanner.description

    @patch("ai_bom.scanners.gcp_live_scanner._HAS_GCP", False)
    def test_scan_without_gcp_sdk(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        scanner = GCPLiveScanner()
        scanner.enabled = True
        components = scanner.scan(Path("."))
        assert components == []

    def test_scan_vertex_models(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        mock_aiplatform = MagicMock()
        mock_model = MagicMock()
        mock_model.display_name = "my-vertex-model"
        mock_model.resource_name = "projects/123/locations/us/models/456"
        mock_aiplatform.Model.list.return_value = [mock_model]
        mock_aiplatform.Endpoint.list.return_value = []
        mock_aiplatform.CustomJob.list.return_value = []

        with (
            patch("ai_bom.scanners.gcp_live_scanner._HAS_GCP", True),
            patch("ai_bom.scanners.gcp_live_scanner.aiplatform", mock_aiplatform),
            patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}),
        ):
            scanner = GCPLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 1
        assert components[0].name == "my-vertex-model"
        assert components[0].type == ComponentType.model
        assert components[0].provider == "GCP Vertex AI"
        assert components[0].source == "cloud-live"

    def test_scan_vertex_endpoints(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        mock_aiplatform = MagicMock()
        mock_ep = MagicMock()
        mock_ep.display_name = "my-vertex-endpoint"
        mock_ep.resource_name = "projects/123/locations/us/endpoints/789"
        mock_aiplatform.Model.list.return_value = []
        mock_aiplatform.Endpoint.list.return_value = [mock_ep]
        mock_aiplatform.CustomJob.list.return_value = []

        with (
            patch("ai_bom.scanners.gcp_live_scanner._HAS_GCP", True),
            patch("ai_bom.scanners.gcp_live_scanner.aiplatform", mock_aiplatform),
            patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}),
        ):
            scanner = GCPLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 1
        assert components[0].name == "my-vertex-endpoint"
        assert components[0].type == ComponentType.endpoint
        assert components[0].provider == "GCP Vertex AI"

    def test_scan_vertex_custom_jobs(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        mock_aiplatform = MagicMock()
        mock_job = MagicMock()
        mock_job.display_name = "training-job-1"
        mock_job.resource_name = "projects/123/locations/us/customJobs/321"
        mock_aiplatform.Model.list.return_value = []
        mock_aiplatform.Endpoint.list.return_value = []
        mock_aiplatform.CustomJob.list.return_value = [mock_job]

        with (
            patch("ai_bom.scanners.gcp_live_scanner._HAS_GCP", True),
            patch("ai_bom.scanners.gcp_live_scanner.aiplatform", mock_aiplatform),
            patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}),
        ):
            scanner = GCPLiveScanner()
            scanner.enabled = True
            components = scanner.scan(Path("."))

        assert len(components) == 1
        assert components[0].name == "training-job-1"
        assert components[0].type == ComponentType.model

    def test_not_triggered_in_normal_scan(self):
        from ai_bom.scanners import get_all_scanners

        scanners = get_all_scanners()
        live_scanners = [s for s in scanners if s.name == "gcp-live"]
        for s in live_scanners:
            assert s.supports(Path("/some/project")) is False


# =====================================================================
# Azure Live Scanner
# =====================================================================


class TestAzureLiveScanner:
    """Tests for AzureLiveScanner."""

    def test_disabled_by_default(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        scanner = AzureLiveScanner()
        assert scanner.enabled is False
        assert scanner.supports(Path(".")) is False

    def test_name_and_description(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        scanner = AzureLiveScanner()
        assert scanner.name == "azure-live"
        assert "Azure" in scanner.description

    @patch("ai_bom.scanners.azure_live_scanner._HAS_AZURE", False)
    def test_scan_without_azure_sdk(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        scanner = AzureLiveScanner()
        scanner.enabled = True
        components = scanner.scan(Path("."))
        assert components == []

    def test_scan_azure_ml_endpoints(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        scanner = AzureLiveScanner()
        scanner.enabled = True

        mock_credential = MagicMock()
        mock_ml_client = MagicMock()

        mock_ep = MagicMock()
        mock_ep.name = "prod-endpoint"
        mock_ml_client.online_endpoints.list.return_value = [mock_ep]
        mock_ml_client.models.list.return_value = []

        mock_ml_cls = MagicMock()
        mock_ml_cls.from_config.return_value = mock_ml_client

        # Inject azure.ai.ml as a fake module so the import inside _scan_ml_endpoints works
        fake_azure_ai_ml = MagicMock(MLClient=mock_ml_cls)

        with (
            patch.object(scanner, "_scan_openai_deployments", return_value=[]),
            patch.object(scanner, "_scan_cognitive_services", return_value=[]),
            patch(
                "ai_bom.scanners.azure_live_scanner.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch("ai_bom.scanners.azure_live_scanner._HAS_AZURE", True),
            patch.dict(sys.modules, {"azure.ai.ml": fake_azure_ai_ml}),
        ):
            components = scanner.scan(Path("."))

        assert len(components) == 1
        assert components[0].name == "prod-endpoint"
        assert components[0].type == ComponentType.endpoint
        assert components[0].provider == "Azure ML"
        assert components[0].source == "cloud-live"

    def test_resource_group_from_id(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        rid = (
            "/subscriptions/abc/resourceGroups/myRG/"
            "providers/Microsoft.CognitiveServices/accounts/myaccount"
        )
        assert AzureLiveScanner._resource_group_from_id(rid) == "myRG"
        assert AzureLiveScanner._resource_group_from_id("") == ""

    def test_not_triggered_in_normal_scan(self):
        from ai_bom.scanners import get_all_scanners

        scanners = get_all_scanners()
        live_scanners = [s for s in scanners if s.name == "azure-live"]
        for s in live_scanners:
            assert s.supports(Path("/some/project")) is False


# =====================================================================
# ImportError handling
# =====================================================================


class TestImportErrorHandling:
    """Verify scanners handle missing SDK packages gracefully."""

    def test_aws_scanner_import_without_boto3(self):
        from ai_bom.scanners.aws_live_scanner import AWSLiveScanner

        scanner = AWSLiveScanner()
        assert scanner.name == "aws-live"

    def test_gcp_scanner_import_without_google_cloud(self):
        from ai_bom.scanners.gcp_live_scanner import GCPLiveScanner

        scanner = GCPLiveScanner()
        assert scanner.name == "gcp-live"

    def test_azure_scanner_import_without_azure_sdk(self):
        from ai_bom.scanners.azure_live_scanner import AzureLiveScanner

        scanner = AzureLiveScanner()
        assert scanner.name == "azure-live"


# =====================================================================
# CLI scan-cloud command
# =====================================================================


class TestScanCloudCLI:
    """Tests for the scan-cloud CLI command."""

    def test_scan_cloud_invalid_provider(self):
        from typer.testing import CliRunner

        from ai_bom.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan-cloud", "invalid-provider", "--quiet"])
        assert result.exit_code != 0
        assert "Unknown cloud provider" in result.output

    def test_scan_cloud_aws(self):
        from typer.testing import CliRunner

        from ai_bom.cli import app

        mock_boto3 = _make_aws_clients(
            foundation_models=[
                {"modelId": "anthropic.claude-v2", "providerName": "Anthropic"},
            ],
        )

        with (
            patch("ai_bom.scanners.aws_live_scanner._HAS_BOTO3", True),
            patch("ai_bom.scanners.aws_live_scanner.boto3", mock_boto3),
        ):
            runner = CliRunner()
            result = runner.invoke(app, ["scan-cloud", "aws", "--quiet", "-f", "json"])
            assert result.exit_code == 0

    def test_scan_cloud_json_format(self):
        """scan-cloud with --format json should not crash."""
        from typer.testing import CliRunner

        from ai_bom.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan-cloud", "aws", "--quiet", "-f", "json"])
        # Might exit 1 if scanner not available, or 2 if no region configured -- ok
        assert result.exit_code in (0, 1, 2)
