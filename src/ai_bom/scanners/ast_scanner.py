"""AST-based Python scanner for deep AI component detection.

Uses Python's ``ast`` module to detect:
- Import statements for known AI packages
- Decorator patterns (@agent, @tool, @crew, @task, @flow)
- Function calls to AI APIs
- String literals containing model names

This scanner is disabled by default and activated via the ``--deep`` CLI flag.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ai_bom.config import CREWAI_FLOW_PATTERNS, KNOWN_AI_PACKAGES, KNOWN_MODEL_PATTERNS
from ai_bom.models import (
    AIComponent,
    ComponentType,
    RiskAssessment,
    SourceLocation,
    UsageType,
)
from ai_bom.scanners.base import BaseScanner

# AI API call patterns: (attribute chain regex, provider, description)
_AI_API_CALLS: list[tuple[str, str, str]] = [
    (r"openai\.ChatCompletion\.create", "OpenAI", "ChatCompletion API"),
    (r"client\.chat\.completions\.create", "OpenAI", "Chat Completions API"),
    (r"client\.messages\.create", "Anthropic", "Messages API"),
    (r"genai\.GenerativeModel", "Google", "Generative AI"),
    (r"ollama\.chat", "Ollama", "Ollama chat"),
    (r"ollama\.generate", "Ollama", "Ollama generate"),
    (r"litellm\.completion", "LiteLLM", "LiteLLM completion"),
    (r"replicate\.run", "Replicate", "Replicate run"),
]


def _attr_chain(node: ast.expr) -> str:
    """Build a dotted attribute string from an AST node.

    For example ``client.chat.completions.create`` yields
    ``"client.chat.completions.create"``.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return ".".join(parts)


class ASTScanner(BaseScanner):
    """Deep AST-based Python analysis scanner.

    Disabled by default.  Enable it by setting ``enabled = True`` before
    calling ``scan()``.
    """

    name = "ast"
    description = "Deep AST-based Python analysis"
    enabled: bool = False

    def supports(self, path: Path) -> bool:
        if not self.enabled:
            return False
        if path.is_dir():
            return True
        return path.suffix == ".py"

    def scan(self, path: Path) -> list[AIComponent]:
        components: list[AIComponent] = []
        for file_path in self.iter_files(path, extensions={".py"}):
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(file_path))
            except (SyntaxError, Exception):
                continue
            components.extend(self._analyse_tree(tree, file_path, source))
        return components

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_tree(self, tree: ast.Module, file_path: Path, source: str) -> list[AIComponent]:
        components: list[AIComponent] = []
        components.extend(self._detect_imports(tree, file_path))
        components.extend(self._detect_decorators(tree, file_path))
        components.extend(self._detect_api_calls(tree, file_path))
        components.extend(self._detect_model_strings(tree, file_path))
        return components

    # -- imports --------------------------------------------------------

    def _detect_imports(self, tree: ast.Module, file_path: Path) -> list[AIComponent]:
        components: list[AIComponent] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            names: list[tuple[str, int]] = []

            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append((node.module, node.lineno))

            for module_name, lineno in names:
                # Check the full module and each prefix
                parts = module_name.split(".")
                for i in range(len(parts), 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in KNOWN_AI_PACKAGES and prefix not in seen:
                        seen.add(prefix)
                        provider, usage = KNOWN_AI_PACKAGES[prefix]
                        usage_enum = self._map_usage(usage)
                        components.append(
                            AIComponent(
                                name=f"{provider} (import: {prefix})",
                                type=ComponentType.llm_provider,
                                provider=provider,
                                location=SourceLocation(
                                    file_path=str(file_path),
                                    line_number=lineno,
                                    context_snippet=f"import {module_name}",
                                ),
                                usage_type=usage_enum,
                                risk=RiskAssessment(),
                                source="ast",
                                metadata={"import": module_name},
                            )
                        )
                        break  # stop at first matching prefix
        return components

    # -- decorators -----------------------------------------------------

    def _detect_decorators(self, tree: ast.Module, file_path: Path) -> list[AIComponent]:
        components: list[AIComponent] = []
        decorator_names = {k.lstrip("@"): v for k, v in CREWAI_FLOW_PATTERNS.items()}

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for dec in node.decorator_list:
                dec_name: str | None = None
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Attribute):
                    dec_name = dec.attr
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        dec_name = dec.func.id
                    elif isinstance(dec.func, ast.Attribute):
                        dec_name = dec.func.attr

                if dec_name and dec_name in decorator_names:
                    pattern_type = decorator_names[dec_name]
                    components.append(
                        AIComponent(
                            name=f"CrewAI {pattern_type}: {getattr(node, 'name', '?')}",
                            type=ComponentType.agent_framework,
                            provider="CrewAI",
                            location=SourceLocation(
                                file_path=str(file_path),
                                line_number=dec.lineno,
                                context_snippet=f"@{dec_name}",
                            ),
                            usage_type=UsageType.agent,
                            risk=RiskAssessment(),
                            source="ast",
                            metadata={"decorator": dec_name, "pattern_type": pattern_type},
                        )
                    )
        return components

    # -- API calls ------------------------------------------------------

    def _detect_api_calls(self, tree: ast.Module, file_path: Path) -> list[AIComponent]:
        components: list[AIComponent] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _attr_chain(node.func)
            if not chain:
                continue
            for pattern_str, provider, desc in _AI_API_CALLS:
                pattern = re.compile(pattern_str)
                if pattern.search(chain) and pattern_str not in seen:
                    seen.add(pattern_str)
                    components.append(
                        AIComponent(
                            name=f"{provider} API call: {desc}",
                            type=ComponentType.llm_provider,
                            provider=provider,
                            location=SourceLocation(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                context_snippet=chain,
                            ),
                            usage_type=UsageType.completion,
                            risk=RiskAssessment(),
                            source="ast",
                            metadata={"call": chain},
                        )
                    )
                    break
        return components

    # -- model strings --------------------------------------------------

    def _detect_model_strings(self, tree: ast.Module, file_path: Path) -> list[AIComponent]:
        components: list[AIComponent] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            for pattern, provider in KNOWN_MODEL_PATTERNS:
                if pattern.fullmatch(value) and value not in seen:
                    seen.add(value)
                    components.append(
                        AIComponent(
                            name=f"Model reference: {value}",
                            type=ComponentType.model,
                            provider=provider,
                            model_name=value,
                            location=SourceLocation(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                context_snippet=f'"{value}"',
                            ),
                            usage_type=UsageType.completion,
                            risk=RiskAssessment(),
                            source="ast",
                            metadata={"model": value},
                        )
                    )
                    break
        return components

    # -- utility --------------------------------------------------------

    @staticmethod
    def _map_usage(usage: str) -> UsageType:
        mapping = {
            "completion": UsageType.completion,
            "embedding": UsageType.embedding,
            "agent": UsageType.agent,
            "tool_use": UsageType.tool_use,
            "orchestration": UsageType.orchestration,
        }
        return mapping.get(usage, UsageType.unknown)
