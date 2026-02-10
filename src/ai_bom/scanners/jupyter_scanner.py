"""Jupyter notebook scanner for AI-BOM: Detects AI components in .ipynb files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ai_bom.config import KNOWN_AI_PACKAGES
from ai_bom.models import AIComponent, ComponentType, SourceLocation, UsageType
from ai_bom.scanners.base import BaseScanner


class JupyterScanner(BaseScanner):
    """Scanner for Jupyter notebooks to detect AI components.

    Detects AI usage in:
    - .ipynb notebook files
    - Import statements in code cells
    - Model loading patterns (AutoModel.from_pretrained, pipeline, etc.)
    - AI library usage

    Parses notebook JSON structure to extract code cells and analyze them.
    """

    name = "jupyter"
    description = "Scan Jupyter notebooks for AI components"

    # Model loading patterns
    MODEL_LOADING_PATTERNS = [
        # HuggingFace Transformers
        (r"AutoModel\.from_pretrained\(['\"]([^'\"]+)", "HuggingFace", "model"),
        (r"AutoTokenizer\.from_pretrained\(['\"]([^'\"]+)", "HuggingFace", "model"),
        (r"AutoModelForCausalLM\.from_pretrained\(['\"]([^'\"]+)", "HuggingFace", "model"),
        (r"AutoModelForSeq2SeqLM\.from_pretrained\(['\"]([^'\"]+)", "HuggingFace", "model"),
        (r"pipeline\(['\"]([^'\"]+)", "HuggingFace", "model"),
        # LangChain
        (r"ChatOpenAI\(", "OpenAI", "llm_provider"),
        (r"OpenAI\(", "OpenAI", "llm_provider"),
        (r"ChatAnthropic\(", "Anthropic", "llm_provider"),
        (r"ChatGoogleGenerativeAI\(", "Google", "llm_provider"),
        (r"Ollama\(", "Ollama", "llm_provider"),
        (r"HuggingFaceHub\(", "HuggingFace", "llm_provider"),
        # OpenAI
        (r"openai\.Client\(", "OpenAI", "llm_provider"),
        (r"openai\.OpenAI\(", "OpenAI", "llm_provider"),
        # Anthropic
        (r"anthropic\.Client\(", "Anthropic", "llm_provider"),
        (r"anthropic\.Anthropic\(", "Anthropic", "llm_provider"),
        # Google
        (r"genai\.GenerativeModel\(['\"]([^'\"]+)", "Google", "model"),
        # Sentence Transformers
        (r"SentenceTransformer\(['\"]([^'\"]+)", "HuggingFace", "model"),
    ]

    def __init__(self) -> None:
        """Initialize the Jupyter scanner."""
        super().__init__()
        # Compile model loading patterns for efficiency
        self._model_patterns = [
            (re.compile(pattern, re.IGNORECASE), provider, comp_type)
            for pattern, provider, comp_type in self.MODEL_LOADING_PATTERNS
        ]

    def supports(self, path: Path) -> bool:
        """Check if this scanner should run on the given path.

        Args:
            path: Directory or file path to check

        Returns:
            True if path is a Jupyter notebook or contains notebooks
        """
        if path.is_file():
            return path.suffix.lower() == ".ipynb"

        # For directories, check if any .ipynb files exist
        if path.is_dir():
            try:
                for file_path in path.rglob("*.ipynb"):
                    if file_path.is_file():
                        return True
            except (OSError, PermissionError):
                pass

        return False

    def scan(self, path: Path) -> list[AIComponent]:
        """Scan Jupyter notebooks for AI components.

        Args:
            path: Directory or file path to scan

        Returns:
            List of detected AI components with metadata
        """
        components: list[AIComponent] = []

        if path.is_file():
            # Scan single notebook file
            if path.suffix.lower() == ".ipynb":
                components.extend(self._scan_notebook(path))
        else:
            # Scan directory for notebook files
            for notebook_file in self.iter_files(path, extensions={".ipynb"}):
                components.extend(self._scan_notebook(notebook_file))

        return components

    def _scan_notebook(self, file_path: Path) -> list[AIComponent]:
        """Parse a Jupyter notebook and extract AI components.

        Args:
            file_path: Path to the notebook file

        Returns:
            List of AI components found in the notebook
        """
        components: list[AIComponent] = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(content)

            if not isinstance(data, dict):
                return components

            cells = data.get("cells", [])
            if not isinstance(cells, list):
                return components

            # Track seen imports to avoid duplicates
            seen_imports: set[str] = set()

            # Iterate through cells
            for cell_idx, cell in enumerate(cells, start=1):
                if not isinstance(cell, dict):
                    continue

                cell_type = cell.get("cell_type", "")
                if cell_type != "code":
                    continue

                # Get source code from cell
                source = cell.get("source", [])
                if isinstance(source, str):
                    source_code = source
                elif isinstance(source, list):
                    source_code = "".join(source)
                else:
                    continue

                # Scan for imports
                for line_num, line in enumerate(source_code.splitlines(), start=1):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Check for import statements
                    import_match = re.match(
                        r"(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", line
                    )
                    if import_match:
                        module_name = import_match.group(1) or import_match.group(2)
                        base_module = module_name.split(".")[0]

                        # Check if it's a known AI package
                        if base_module in KNOWN_AI_PACKAGES or module_name in KNOWN_AI_PACKAGES:
                            package_key = (
                                module_name if module_name in KNOWN_AI_PACKAGES else base_module
                            )

                            # Avoid duplicates
                            if package_key in seen_imports:
                                continue
                            seen_imports.add(package_key)

                            provider, usage_type = KNOWN_AI_PACKAGES[package_key]

                            # Map usage type string to enum
                            usage_enum = self._map_usage_type(usage_type)

                            # Determine component type
                            comp_type = self._determine_component_type(package_key, usage_type)

                            component = AIComponent(
                                name=package_key,
                                type=comp_type,
                                version="",
                                provider=provider,
                                location=SourceLocation(
                                    file_path=str(file_path.resolve()),
                                    line_number=None,
                                    context_snippet=(
                                        f"Cell {cell_idx}, Line {line_num}: {line[:80]}"
                                    ),
                                ),
                                usage_type=usage_enum,
                                source="jupyter",
                                metadata={
                                    "cell_number": cell_idx,
                                    "line_in_cell": line_num,
                                    "import_statement": line,
                                },
                            )
                            components.append(component)

                # Check for model loading patterns
                components.extend(self._check_model_loading(source_code, file_path, cell_idx))

        except json.JSONDecodeError:
            # Invalid JSON, skip this file
            pass
        except (OSError, UnicodeDecodeError):
            # File read error, skip
            pass

        return components

    def _check_model_loading(
        self, source_code: str, file_path: Path, cell_idx: int
    ) -> list[AIComponent]:
        """Check for model loading patterns in code.

        Args:
            source_code: Source code from notebook cell
            file_path: Path to notebook file
            cell_idx: Cell index number

        Returns:
            List of AI components representing model loading
        """
        components: list[AIComponent] = []
        seen_models: set[str] = set()

        for pattern, provider, comp_type_str in self._model_patterns:
            for match in pattern.finditer(source_code):
                # Extract model name if captured
                model_name = match.group(1) if match.groups() else ""

                # Create unique key for deduplication
                key = f"{provider}:{comp_type_str}:{model_name}"
                if key in seen_models:
                    continue
                seen_models.add(key)

                # Determine component type
                if comp_type_str == "model":
                    comp_type = ComponentType.model
                    usage_type = UsageType.completion
                else:
                    comp_type = ComponentType.llm_provider
                    usage_type = UsageType.completion

                # Get context snippet
                match_line = source_code[: match.start()].count("\n") + 1
                context_snippet = f"Cell {cell_idx}, Model loading: {match.group(0)[:60]}"

                component = AIComponent(
                    name=model_name or f"{provider} Model",
                    type=comp_type,
                    version="",
                    provider=provider,
                    model_name=model_name,
                    location=SourceLocation(
                        file_path=str(file_path.resolve()),
                        line_number=None,
                        context_snippet=context_snippet,
                    ),
                    usage_type=usage_type,
                    source="jupyter",
                    metadata={
                        "cell_number": cell_idx,
                        "line_in_cell": match_line,
                        "pattern_matched": match.group(0),
                    },
                )
                components.append(component)

        return components

    def _map_usage_type(self, usage_type_str: str) -> UsageType:
        """Map usage type string to enum.

        Args:
            usage_type_str: Usage type string from config

        Returns:
            UsageType enum value
        """
        mapping = {
            "completion": UsageType.completion,
            "embedding": UsageType.embedding,
            "orchestration": UsageType.orchestration,
            "agent": UsageType.agent,
            "tool_use": UsageType.tool_use,
        }
        return mapping.get(usage_type_str, UsageType.unknown)

    def _determine_component_type(self, package_name: str, usage_type: str) -> ComponentType:
        """Determine component type from package name and usage type.

        Args:
            package_name: Name of the package
            usage_type: Usage type string

        Returns:
            ComponentType enum value
        """
        # Map to component types
        if usage_type == "agent":
            return ComponentType.agent_framework
        if usage_type == "orchestration":
            return ComponentType.agent_framework
        if usage_type == "embedding":
            return ComponentType.llm_provider
        if "transformers" in package_name.lower():
            return ComponentType.model

        return ComponentType.llm_provider
