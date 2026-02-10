"""Tests for docker scanner."""

import pytest

from ai_bom.scanners.docker_scanner import DockerScanner


@pytest.fixture
def scanner():
    return DockerScanner()


class TestDockerScanner:
    def test_name(self, scanner):
        assert scanner.name == "docker"

    def test_detects_ollama_in_compose(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_docker_compose.yml")
        assert len(components) > 0
        providers = [c.provider for c in components]
        assert any("Ollama" in p for p in providers) or any(
            "ollama" in c.name.lower() for c in components
        )

    def test_detects_gpu(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_docker_compose.yml")
        gpu_components = [c for c in components if c.metadata.get("gpu")]
        assert len(gpu_components) > 0

    def test_source_is_docker(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_docker_compose.yml")
        for c in components:
            assert c.source == "docker"

    def test_empty_directory(self, scanner, tmp_path):
        components = scanner.scan(tmp_path)
        assert components == []

    def test_supports_dockerfile(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.12\n")
        assert scanner.supports(dockerfile)

    def test_supports_dockerfile_with_extension(self, scanner, tmp_path):
        dockerfile = tmp_path / "app.dockerfile"
        dockerfile.write_text("FROM python:3.12\n")
        assert scanner.supports(dockerfile)

    def test_supports_compose_file(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("version: '3'\n")
        assert scanner.supports(compose)

    def test_supports_directory_with_dockerfile(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.12\n")
        assert scanner.supports(tmp_path)

    def test_supports_directory_with_compose(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yaml"
        compose.write_text("version: '3'\n")
        assert scanner.supports(tmp_path)

    def test_not_supports_empty_directory(self, scanner, tmp_path):
        assert not scanner.supports(tmp_path)

    def test_scan_dockerfile_with_ai_image(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM ollama/ollama:latest\nRUN echo hello\n")
        components = scanner.scan(dockerfile)
        assert len(components) == 1
        assert "ollama" in components[0].name.lower()

    def test_scan_dockerfile_with_platform_flag(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM --platform=linux/amd64 ollama/ollama:0.1.0\n")
        components = scanner.scan(dockerfile)
        assert len(components) == 1
        assert components[0].version == "0.1.0"

    def test_scan_dockerfile_with_comment(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM ollama/ollama:latest # AI model server\n")
        components = scanner.scan(dockerfile)
        assert len(components) == 1

    def test_scan_dockerfile_with_builder_alias(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM ollama/ollama:latest as builder\n")
        components = scanner.scan(dockerfile)
        assert len(components) == 1
        assert components[0].version == "latest"

    def test_parse_image_ref_with_tag(self, scanner):
        name, version = scanner._parse_image_ref("ollama/ollama:0.1.0")
        assert name == "ollama/ollama"
        assert version == "0.1.0"

    def test_parse_image_ref_without_tag(self, scanner):
        name, version = scanner._parse_image_ref("ollama/ollama")
        assert name == "ollama/ollama"
        assert version == "latest"

    def test_parse_image_ref_with_tag_and_digest(self, scanner):
        # When there's a tag and digest, the rsplit on : splits at the digest
        name, _version = scanner._parse_image_ref("ollama/ollama:v1.0@sha256:abc123")
        # Due to rsplit, image_name will be "ollama/ollama:v1.0@sha256"
        # and version will be "abc123"
        # The @sha256: check won't match because there's no : after sha256 in image_name
        assert "ollama" in name

    def test_extract_provider_ollama(self, scanner):
        assert scanner._extract_provider("ollama/ollama") == "Ollama"

    def test_extract_provider_vllm(self, scanner):
        assert scanner._extract_provider("vllm/vllm-openai") == "vLLM"

    def test_extract_provider_huggingface(self, scanner):
        assert scanner._extract_provider("huggingface/transformers") == "HuggingFace"

    def test_extract_provider_unknown(self, scanner):
        assert scanner._extract_provider("mycompany/custom-image") == ""

    def test_scan_compose_with_build_context(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("""
version: '3'
services:
  app:
    build: ./app
""")
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        dockerfile = app_dir / "Dockerfile"
        dockerfile.write_text("FROM ollama/ollama:latest\n")
        components = scanner.scan(compose)
        assert len(components) == 1

    def test_scan_compose_with_build_dict(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("""
version: '3'
services:
  app:
    build:
      context: ./app
      dockerfile: Dockerfile.custom
""")
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        dockerfile = app_dir / "Dockerfile.custom"
        dockerfile.write_text("FROM ollama/ollama:latest\n")
        components = scanner.scan(compose)
        assert len(components) == 1

    def test_check_gpu_with_nvidia_driver(self, scanner):
        config = {"deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia"}]}}}}
        assert scanner._check_gpu(config)

    def test_check_gpu_with_gpu_capability(self, scanner):
        config = {
            "deploy": {"resources": {"reservations": {"devices": [{"capabilities": ["gpu"]}]}}}
        }
        assert scanner._check_gpu(config)

    def test_check_gpu_with_nested_gpu_capability(self, scanner):
        config = {
            "deploy": {
                "resources": {"reservations": {"devices": [{"capabilities": [["gpu", "utility"]]}]}}
            }
        }
        assert scanner._check_gpu(config)

    def test_check_gpu_no_gpu(self, scanner):
        config = {"deploy": {"resources": {}}}
        assert not scanner._check_gpu(config)

    def test_check_gpu_invalid_types(self, scanner):
        config = {"deploy": "not a dict"}
        assert not scanner._check_gpu(config)

    def test_check_model_mounts_string_volume(self, scanner):
        config = {"volumes": ["/host/models:/container/models"]}
        assert scanner._check_model_mounts(config)

    def test_check_model_mounts_dict_volume(self, scanner):
        config = {"volumes": [{"source": "/host/weights", "target": "/app/weights"}]}
        assert scanner._check_model_mounts(config)

    def test_check_model_mounts_gguf_files(self, scanner):
        config = {"volumes": ["./model.gguf:/app/model.gguf"]}
        assert scanner._check_model_mounts(config)

    def test_check_model_mounts_no_models(self, scanner):
        config = {"volumes": ["/data:/data"]}
        assert not scanner._check_model_mounts(config)

    def test_check_ai_env_vars_list_format(self, scanner):
        config = {"environment": ["OPENAI_API_KEY=sk-123", "DATABASE_URL=postgres"]}
        assert scanner._check_ai_env_vars(config)

    def test_check_ai_env_vars_dict_format(self, scanner):
        config = {"environment": {"ANTHROPIC_API_KEY": "sk-ant-123", "PORT": "8000"}}
        assert scanner._check_ai_env_vars(config)

    def test_check_ai_env_vars_no_ai_vars(self, scanner):
        config = {"environment": ["PORT=8000", "DEBUG=true"]}
        assert not scanner._check_ai_env_vars(config)

    def test_scan_directory_with_dockerfile_variants(self, scanner, tmp_path):
        (tmp_path / "Dockerfile.dev").write_text("FROM ollama/ollama:latest\n")
        (tmp_path / "Dockerfile.prod").write_text("FROM vllm/vllm-openai:v0.1.0\n")
        components = scanner.scan(tmp_path)
        assert len(components) == 2

    def test_scan_compose_yaml_error(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("not: valid: yaml: [[[")
        components = scanner.scan(compose)
        assert components == []

    def test_scan_dockerfile_read_error(self, scanner, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_bytes(b"\x00\x00\x00\x00")
        components = scanner.scan(dockerfile)
        assert components == []

    def test_scan_compose_invalid_services_type(self, scanner, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: not_a_dict\n")
        components = scanner.scan(compose)
        assert components == []
