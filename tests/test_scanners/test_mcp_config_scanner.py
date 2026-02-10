"""Tests for MCP configuration scanner."""

import json

import pytest

from ai_bom.models import ComponentType, UsageType
from ai_bom.scanners.mcp_config_scanner import MCPConfigScanner


@pytest.fixture
def scanner():
    """Create an MCPConfigScanner instance."""
    return MCPConfigScanner()


def test_scanner_registration():
    """Test that scanner is properly registered."""
    scanner = MCPConfigScanner()
    assert scanner.name == "mcp-config"
    assert scanner.description == "Scan MCP configuration files for AI server declarations"


def test_supports_mcp_json(tmp_path, scanner):
    """Test that scanner supports mcp.json file."""
    config_file = tmp_path / "mcp.json"
    config_file.write_text("{}")

    assert scanner.supports(config_file)


def test_supports_claude_desktop_config(tmp_path, scanner):
    """Test that scanner supports claude_desktop_config.json."""
    config_file = tmp_path / "claude_desktop_config.json"
    config_file.write_text("{}")

    assert scanner.supports(config_file)


def test_supports_cursor_mcp_config(tmp_path, scanner):
    """Test that scanner supports .cursor/mcp.json."""
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    config_file = cursor_dir / "mcp.json"
    config_file.write_text("{}")

    assert scanner.supports(config_file)


def test_supports_directory_with_mcp_config(tmp_path, scanner):
    """Test that scanner supports directories with MCP config files."""
    config_file = tmp_path / "mcp.json"
    config_file.write_text("{}")

    assert scanner.supports(tmp_path)


def test_not_supports_non_mcp_file(tmp_path, scanner):
    """Test that scanner does not support non-MCP files."""
    test_file = tmp_path / "test.json"
    test_file.write_text("{}")

    assert not scanner.supports(test_file)


def test_scan_claude_desktop_config_structure(tmp_path, scanner):
    """Test scanning Claude Desktop config structure."""
    config_file = tmp_path / "claude_desktop_config.json"

    config_data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.type == ComponentType.mcp_server
    assert "filesystem" in comp.name.lower()
    assert comp.metadata["command"] == "npx"
    assert comp.metadata["transport"] == "stdio"
    assert comp.source == "mcp-config"


def test_scan_mcp_json_structure(tmp_path, scanner):
    """Test scanning standalone mcp.json structure."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "brave-search": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": "your-key-here"},
            }
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert "brave-search" in comp.name.lower()
    assert "env_vars" in comp.metadata
    assert "BRAVE_API_KEY" in comp.metadata["env_vars"]


def test_scan_nested_mcp_structure(tmp_path, scanner):
    """Test scanning nested mcp config structure."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "mcp": {
            "servers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                }
            }
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert "github" in comp.name.lower()


def test_scan_multiple_servers(tmp_path, scanner):
    """Test scanning config with multiple MCP servers."""
    config_file = tmp_path / "claude_desktop_config.json"

    config_data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
            "brave-search": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            },
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 3
    server_names = {c.metadata["server_name"] for c in components}
    assert "filesystem" in server_names
    assert "github" in server_names
    assert "brave-search" in server_names


def test_scan_server_with_transport(tmp_path, scanner):
    """Test scanning server with explicit transport type."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "remote-server": {
                "command": "node",
                "args": ["server.js"],
                "transport": "sse",
                "url": "http://localhost:3000",
            }
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.metadata["transport"] == "sse"


def test_scan_server_with_url_sse_transport(tmp_path, scanner):
    """Test that URL presence implies SSE transport."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "remote": {
                "command": "node",
                "url": "http://example.com",
            }
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.metadata["transport"] == "sse"


def test_provider_guessing_filesystem(tmp_path, scanner):
    """Test provider guessing for filesystem server."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "filesystem": {"command": "npx", "args": ["@modelcontextprotocol/server-filesystem"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.provider == "MCP Filesystem"


def test_provider_guessing_github(tmp_path, scanner):
    """Test provider guessing for GitHub server."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "github": {"command": "npx", "args": ["@modelcontextprotocol/server-github"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.provider == "GitHub"


def test_provider_guessing_brave(tmp_path, scanner):
    """Test provider guessing for Brave Search server."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "brave-search": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-brave-search"],
            },
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.provider == "Brave Search"


def test_trusted_command_detection(tmp_path, scanner):
    """Test detection of trusted vs untrusted commands."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "trusted": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            },
            "untrusted": {"command": "/tmp/sketchy-binary", "args": []},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 2

    # Find untrusted component
    untrusted = next(c for c in components if c.metadata["server_name"] == "untrusted")
    assert "mcp_unknown_server" in untrusted.flags

    # Find trusted component
    trusted = next(c for c in components if c.metadata["server_name"] == "trusted")
    assert "mcp_unknown_server" not in trusted.flags


def test_internet_facing_flag(tmp_path, scanner):
    """Test flagging of internet-facing servers."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "http-fetch": {"command": "npx", "args": ["@modelcontextprotocol/server-fetch"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert "internet_facing" in comp.flags


def test_usage_type_tool_use(tmp_path, scanner):
    """Test that MCP servers have tool_use usage type."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "test": {"command": "npx", "args": ["test"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.usage_type == UsageType.tool_use


def test_scan_invalid_json(tmp_path, scanner):
    """Test scanning invalid JSON file."""
    config_file = tmp_path / "mcp.json"
    config_file.write_text("{ invalid json [")

    components = scanner.scan(tmp_path)
    # Should not crash, just return empty
    assert len(components) == 0


def test_scan_empty_config(tmp_path, scanner):
    """Test scanning empty config file."""
    config_file = tmp_path / "mcp.json"
    config_file.write_text("{}")

    components = scanner.scan(tmp_path)
    assert len(components) == 0


def test_scan_server_without_command(tmp_path, scanner):
    """Test scanning server entry without command (invalid)."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "invalid": {"args": ["test"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)
    # Should skip invalid server
    assert len(components) == 0


def test_scan_cline_mcp_settings(tmp_path, scanner):
    """Test scanning cline_mcp_settings.json file."""
    config_file = tmp_path / "cline_mcp_settings.json"

    config_data = {
        "mcpServers": {
            "test-server": {"command": "npx", "args": ["test"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1


def test_scan_cursor_directory(tmp_path, scanner):
    """Test scanning .cursor/mcp.json in directory."""
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    config_file = cursor_dir / "mcp.json"

    config_data = {
        "servers": {
            "cursor-server": {"command": "node", "args": ["server.js"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert "cursor-server" in comp.name.lower()


def test_metadata_includes_args(tmp_path, scanner):
    """Test that metadata includes args array."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "test": {"command": "npx", "args": ["-y", "test-server", "--port", "3000"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert "args" in comp.metadata
    assert comp.metadata["args"] == ["-y", "test-server", "--port", "3000"]


def test_provider_guessing_database(tmp_path, scanner):
    """Test provider guessing for database servers."""
    config_file = tmp_path / "mcp.json"

    config_data = {
        "servers": {
            "sqlite": {"command": "npx", "args": ["@modelcontextprotocol/server-sqlite"]},
        }
    }

    config_file.write_text(json.dumps(config_data))

    components = scanner.scan(tmp_path)

    assert len(components) == 1
    comp = components[0]
    assert comp.provider == "MCP Database"
