"""MCP configuration scanner for AI-BOM: Detects MCP server declarations in config files."""

from __future__ import annotations

import json
from pathlib import Path

from ai_bom.config import MCP_CONFIG_FILES
from ai_bom.models import AIComponent, ComponentType, SourceLocation, UsageType
from ai_bom.scanners.base import BaseScanner


class MCPConfigScanner(BaseScanner):
    """Scanner for MCP (Model Context Protocol) configuration files.

    Detects MCP server declarations in:
    - mcp.json
    - .mcp.json
    - mcp-config.json
    - claude_desktop_config.json
    - cline_mcp_settings.json
    - .cursor/mcp.json

    Parses JSON configuration to extract:
    - Server names
    - Command paths
    - Transport types (stdio, sse)
    - Environment variables
    """

    name = "mcp-config"
    description = "Scan MCP configuration files for AI server declarations"

    # Additional config file patterns to check
    ADDITIONAL_CONFIG_PATHS = [
        ".cursor/mcp.json",
        "cline_mcp_settings.json",
    ]

    def supports(self, path: Path) -> bool:
        """Check if this scanner should run on the given path.

        Args:
            path: Directory or file path to check

        Returns:
            True if path is an MCP config file or contains MCP config files
        """
        if path.is_file():
            # Check if file is a known MCP config file
            return path.name in MCP_CONFIG_FILES or (
                path.name == "mcp.json" and path.parent.name == ".cursor"
            )

        # For directories, check if any MCP config files exist
        if path.is_dir():
            try:
                # Check for known config files
                for config_file in MCP_CONFIG_FILES:
                    if (path / config_file).exists():
                        return True
                # Check additional paths
                for config_path in self.ADDITIONAL_CONFIG_PATHS:
                    if (path / config_path).exists():
                        return True
            except (OSError, PermissionError):
                pass

        return False

    def scan(self, path: Path) -> list[AIComponent]:
        """Scan MCP configuration files for server declarations.

        Args:
            path: Directory or file path to scan

        Returns:
            List of detected MCP server components with metadata
        """
        components: list[AIComponent] = []

        if path.is_file():
            # Scan single config file
            if self.supports(path):
                components.extend(self._scan_config_file(path))
        else:
            # Scan directory for config files
            # Check known config files
            for config_file in MCP_CONFIG_FILES:
                config_path = path / config_file
                if config_path.exists() and config_path.is_file():
                    components.extend(self._scan_config_file(config_path))

            # Check additional config paths
            for config_path_str in self.ADDITIONAL_CONFIG_PATHS:
                config_path = path / config_path_str
                if config_path.exists() and config_path.is_file():
                    components.extend(self._scan_config_file(config_path))

        return components

    def _scan_config_file(self, file_path: Path) -> list[AIComponent]:
        """Parse an MCP configuration file and extract server declarations.

        Args:
            file_path: Path to the config file

        Returns:
            List of MCP server components found in the config
        """
        components: list[AIComponent] = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(content)

            if not isinstance(data, dict):
                return components

            # Look for MCP servers in different config structures
            # Structure 1: { "mcpServers": { ... } } (claude_desktop_config.json)
            servers = data.get("mcpServers", {})

            # Structure 2: { "mcp": { "servers": { ... } } }
            if not servers and "mcp" in data:
                mcp_config = data.get("mcp", {})
                if isinstance(mcp_config, dict):
                    servers = mcp_config.get("servers", {})

            # Structure 3: { "servers": { ... } } (standalone mcp.json)
            if not servers and "servers" in data:
                servers = data.get("servers", {})

            if not isinstance(servers, dict):
                return components

            # Parse each server entry
            for server_name, server_config in servers.items():
                if not isinstance(server_config, dict):
                    continue

                component = self._create_server_component(server_name, server_config, file_path)
                if component:
                    components.append(component)

        except json.JSONDecodeError:
            # Invalid JSON, skip this file
            pass
        except (OSError, UnicodeDecodeError):
            # File read error, skip
            pass

        return components

    def _create_server_component(
        self, server_name: str, server_config: dict, file_path: Path
    ) -> AIComponent | None:
        """Create an AIComponent from an MCP server configuration.

        Args:
            server_name: Name of the MCP server
            server_config: Server configuration dictionary
            file_path: Path to config file

        Returns:
            AIComponent or None if invalid configuration
        """
        # Extract command (required)
        command = server_config.get("command")
        if not command:
            return None

        # Extract additional metadata
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        # Determine transport type
        transport = "stdio"  # Default
        if "transport" in server_config:
            transport = str(server_config.get("transport", "stdio"))
        elif "url" in server_config:
            transport = "sse"

        # Determine provider from server name or command
        provider = self._guess_provider(server_name, command)

        # Build metadata
        metadata: dict = {
            "server_name": server_name,
            "command": command,
            "transport": transport,
        }

        if args:
            metadata["args"] = args if isinstance(args, list) else [str(args)]

        if env:
            metadata["env_vars"] = list(env.keys()) if isinstance(env, dict) else []

        # Check for security flags
        flags: list[str] = []

        # Flag if server uses unknown/untrusted command
        if not self._is_trusted_command(command):
            flags.append("mcp_unknown_server")

        # Flag if server has internet access capabilities
        if "http" in server_name.lower() or "fetch" in server_name.lower():
            flags.append("internet_facing")

        # Create component
        component = AIComponent(
            name=f"{server_name} (MCP Server)",
            type=ComponentType.mcp_server,
            version="",
            provider=provider,
            location=SourceLocation(
                file_path=str(file_path.resolve()),
                line_number=None,
                context_snippet=f"MCP Server: {server_name}",
            ),
            usage_type=UsageType.tool_use,
            source="mcp-config",
            metadata=metadata,
            flags=flags,
        )

        return component

    def _guess_provider(self, server_name: str, command: str) -> str:
        """Guess provider from server name or command.

        Args:
            server_name: Name of the MCP server
            command: Command path

        Returns:
            Provider name
        """
        combined = f"{server_name} {command}".lower()

        # Check for known providers (check more specific patterns first)
        if "anthropic" in combined or "claude" in combined:
            return "Anthropic"
        if "openai" in combined:
            return "OpenAI"
        if "google" in combined or "gemini" in combined:
            return "Google"
        if "filesystem" in combined or "fs" in server_name.lower():
            return "MCP Filesystem"
        if "github" in combined:  # Check github before git
            return "GitHub"
        if "git" in combined:
            return "MCP Git"
        if "sqlite" in combined or "postgres" in combined or "database" in combined:
            return "MCP Database"
        if "fetch" in combined or "http" in combined:
            return "MCP Fetch"
        if "brave" in combined:
            return "Brave Search"
        if "puppeteer" in combined or "browser" in combined:
            return "MCP Browser"
        if "slack" in combined:
            return "Slack"
        if "memory" in combined:
            return "MCP Memory"
        if "time" in combined:
            return "MCP Time"

        return "MCP"

    def _is_trusted_command(self, command: str) -> bool:
        """Check if command is from a trusted source.

        Args:
            command: Command path or executable

        Returns:
            True if command is trusted, False otherwise
        """
        trusted_patterns = [
            "npx",
            "node",
            "python",
            "python3",
            "uvx",
            "mcp-server-",
            "@modelcontextprotocol/",
            "@anthropic/",
        ]

        command_lower = command.lower()
        return any(pattern in command_lower for pattern in trusted_patterns)
