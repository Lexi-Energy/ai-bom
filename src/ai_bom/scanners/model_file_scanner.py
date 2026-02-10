"""Model file scanner for AI-BOM: Detects AI model binary files."""

from __future__ import annotations

import os
from pathlib import Path

from ai_bom.models import AIComponent, ComponentType, SourceLocation, UsageType
from ai_bom.scanners.base import BaseScanner


class ModelFileScanner(BaseScanner):
    """Scanner for AI model binary files.

    Detects AI model files by extension:
    - .onnx (ONNX models)
    - .pt, .pth (PyTorch models)
    - .pb (TensorFlow models)
    - .tflite (TensorFlow Lite models)
    - .mlmodel (Core ML models)
    - .safetensors (Safetensors format)
    - .gguf, .ggml (llama.cpp quantized models)
    - .bin files >1MB (heuristic for model weights)

    Reports file size, format, and location for each detected model.
    """

    name = "model-files"
    description = "Detect AI model binary files"

    # Model file extensions and their associated formats/frameworks
    MODEL_EXTENSIONS = {
        ".onnx": ("ONNX", "ONNX Runtime"),
        ".pt": ("PyTorch", "PyTorch"),
        ".pth": ("PyTorch", "PyTorch"),
        ".pb": ("TensorFlow", "TensorFlow"),
        ".tflite": ("TensorFlow Lite", "TensorFlow"),
        ".mlmodel": ("Core ML", "Apple Core ML"),
        ".safetensors": ("Safetensors", "HuggingFace"),
        ".gguf": ("GGUF", "llama.cpp"),
        ".ggml": ("GGML", "llama.cpp"),
    }

    # Minimum size for .bin files to be considered models (1MB)
    MIN_BIN_SIZE = 1_048_576

    def supports(self, path: Path) -> bool:
        """Check if this scanner should run on the given path.

        Args:
            path: Directory or file path to check

        Returns:
            True if path is a model file or directory containing model files
        """
        if path.is_file():
            return self._is_model_file(path)

        # For directories, check if any model files exist
        if path.is_dir():
            try:
                # Quick check for any model extensions
                for ext in self.MODEL_EXTENSIONS:
                    if any(path.rglob(f"*{ext}")):
                        return True
                # Check for large .bin files
                for bin_file in path.rglob("*.bin"):
                    if bin_file.is_file() and self._is_model_file(bin_file):
                        return True
            except (OSError, PermissionError):
                pass

        return False

    def scan(self, path: Path) -> list[AIComponent]:
        """Scan for AI model binary files.

        Args:
            path: Directory or file path to scan

        Returns:
            List of detected AI model components with metadata
        """
        components: list[AIComponent] = []

        if path.is_file():
            # Scan single file
            if self._is_model_file(path):
                component = self._create_component_from_file(path)
                if component:
                    components.append(component)
        else:
            # Scan directory for model files
            # First, scan known extensions
            for ext in self.MODEL_EXTENSIONS:
                for model_file in path.rglob(f"*{ext}"):
                    if model_file.is_file():
                        component = self._create_component_from_file(model_file)
                        if component:
                            components.append(component)

            # Then scan .bin files >1MB
            for bin_file in path.rglob("*.bin"):
                if bin_file.is_file() and self._is_model_file(bin_file):
                    component = self._create_component_from_file(bin_file)
                    if component:
                        components.append(component)

        return components

    def _is_model_file(self, file_path: Path) -> bool:
        """Check if a file is likely a model file.

        Args:
            file_path: Path to file to check

        Returns:
            True if file is likely a model file
        """
        ext = file_path.suffix.lower()

        # Check known model extensions
        if ext in self.MODEL_EXTENSIONS:
            return True

        # For .bin files, check size heuristic
        if ext == ".bin":
            try:
                file_size = os.path.getsize(file_path)
                return file_size >= self.MIN_BIN_SIZE
            except OSError:
                return False

        return False

    def _create_component_from_file(self, file_path: Path) -> AIComponent | None:
        """Create an AIComponent from a model file.

        Args:
            file_path: Path to model file

        Returns:
            AIComponent or None if file cannot be read
        """
        try:
            file_size = os.path.getsize(file_path)
            ext = file_path.suffix.lower()

            # Determine format and provider
            if ext in self.MODEL_EXTENSIONS:
                format_name, provider = self.MODEL_EXTENSIONS[ext]
            elif ext == ".bin":
                format_name = "Binary Model Weights"
                provider = self._guess_provider_from_path(file_path)
            else:
                return None

            # Create component
            component = AIComponent(
                name=file_path.name,
                type=ComponentType.model,
                version="",
                provider=provider,
                location=SourceLocation(
                    file_path=str(file_path.resolve()),
                    line_number=None,
                    context_snippet=f"Model file: {file_path.name}",
                ),
                usage_type=UsageType.unknown,
                source="model-file",
                metadata={
                    "file_size_bytes": file_size,
                    "format": format_name,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "extension": ext,
                },
            )

            # Add flags for large models
            if file_size > 1_073_741_824:  # 1GB
                component.flags.append("large_model_file")

            return component

        except (OSError, PermissionError):
            return None

    def _guess_provider_from_path(self, file_path: Path) -> str:
        """Guess provider/framework from file path.

        Args:
            file_path: Path to model file

        Returns:
            Provider name or "Unknown"
        """
        path_str = str(file_path).lower()

        # Check for common patterns in path
        if "huggingface" in path_str or "hf" in path_str:
            return "HuggingFace"
        if "pytorch" in path_str or "torch" in path_str:
            return "PyTorch"
        if "tensorflow" in path_str or "tf" in path_str:
            return "TensorFlow"
        if "onnx" in path_str:
            return "ONNX Runtime"
        if "llama" in path_str:
            return "llama.cpp"
        if "safetensors" in path_str:
            return "HuggingFace"
        if "mistral" in path_str:
            return "Mistral"
        if "openai" in path_str:
            return "OpenAI"
        if "anthropic" in path_str or "claude" in path_str:
            return "Anthropic"
        if "google" in path_str or "gemini" in path_str:
            return "Google"

        return "Unknown"
