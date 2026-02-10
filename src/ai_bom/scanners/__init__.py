"""Scanner modules for AI-BOM detection.

Importing this package triggers auto-registration of all scanner classes
via the __init_subclass__ hook in BaseScanner.

Available scanners:
    - CodeScanner: Detects AI libraries and frameworks in source code
    - DockerScanner: Detects AI services in Docker/Kubernetes deployments
    - NetworkScanner: Detects AI endpoints and credentials in config files
    - CloudScanner: Detects AI services in Terraform, CloudFormation, etc.
    - N8nScanner: Detects AI components in n8n workflow automation
    - ASTScanner: Deep AST-based Python analysis (--deep flag)
    - GitHubActionsScanner: Detects AI components in GitHub Actions workflows
    - JupyterScanner: Detects AI components in Jupyter notebooks
    - ModelFileScanner: Detects AI model binary files
    - MCPConfigScanner: Detects MCP server configurations
    - AWSLiveScanner: Scans live AWS account for AI/ML services (optional)
    - GCPLiveScanner: Scans live GCP project for AI/ML services (optional)
    - AzureLiveScanner: Scans live Azure subscription for AI/ML services (optional)
"""

from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import scanner modules to trigger registration via __init_subclass__
from ai_bom.scanners import (  # noqa: F401
    ast_scanner,
    cloud_scanner,
    code_scanner,
    docker_scanner,
    github_actions_scanner,
    jupyter_scanner,
    mcp_config_scanner,
    model_file_scanner,
    n8n_scanner,
    network_scanner,
)
from ai_bom.scanners.base import BaseScanner, get_all_scanners

# Live cloud scanners — optional dependencies, skip if SDK not installed
with contextlib.suppress(ImportError):
    from ai_bom.scanners import aws_live_scanner  # noqa: F401

with contextlib.suppress(ImportError):
    from ai_bom.scanners import gcp_live_scanner  # noqa: F401

with contextlib.suppress(ImportError):
    from ai_bom.scanners import azure_live_scanner  # noqa: F401

logger = logging.getLogger(__name__)


def run_scanners_parallel(
    scanners: list[BaseScanner],
    path: Path,
    workers: int = 4,
) -> list:
    """Run multiple scanners in parallel using a thread pool.

    Only scanners whose ``supports(path)`` returns True are executed.
    Exceptions in individual scanners are logged and do not abort the
    remaining work.

    Args:
        scanners: List of scanner instances to run.
        path: Target path to scan.
        workers: Maximum number of concurrent workers.

    Returns:
        Flat list of :class:`AIComponent` objects from all scanners.
    """
    from ai_bom.models import AIComponent  # local to avoid circular

    results: list[AIComponent] = []

    supported = [s for s in scanners if s.supports(path)]
    if not supported:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_scanner = {executor.submit(s.scan, path): s for s in supported}
        for future in as_completed(future_to_scanner):
            scanner = future_to_scanner[future]
            try:
                components = future.result()
                results.extend(components)
            except Exception:
                logger.exception("Error in parallel scanner %s", scanner.name)

    return results


__all__ = ["BaseScanner", "get_all_scanners", "run_scanners_parallel"]
