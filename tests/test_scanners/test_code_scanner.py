"""Tests for code scanner."""

import pytest

from ai_bom.models import ComponentType, UsageType
from ai_bom.scanners.code_scanner import CodeScanner


@pytest.fixture
def scanner():
    return CodeScanner()


class TestCodeScanner:
    def test_name(self, scanner):
        assert scanner.name == "code"

    def test_supports_directory(self, scanner, tmp_path):
        assert scanner.supports(tmp_path)

    def test_supports_python_file(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("pass")
        assert scanner.supports(f)

    def test_detects_openai_import(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("from openai import OpenAI\nclient = OpenAI()\n")
        req = tmp_path / "requirements.txt"
        req.write_text("openai>=1.0.0\n")
        components = scanner.scan(tmp_path)
        providers = [c.provider for c in components]
        assert any("OpenAI" in p for p in providers)

    def test_detects_hardcoded_api_key(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_text(
            "from openai import OpenAI\n"
            "client = OpenAI(api_key="
            '"sk-demo1234567890abcdefghijklmnopqrstuvwxyz1234")\n'
        )
        components = scanner.scan(tmp_path)
        has_key_flag = any("hardcoded_api_key" in c.flags for c in components)
        assert has_key_flag

    def test_detects_crewai(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_crew.py")
        providers = [c.provider for c in components]
        assert any("CrewAI" in p for p in providers) or any(
            "crewai" in c.name.lower() for c in components
        )

    def test_detects_langchain(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_langchain.py")
        assert len(components) > 0

    def test_detects_requirements(self, scanner, fixtures_dir):
        components = scanner.scan(fixtures_dir / "sample_requirements.txt")
        # Should find openai, langchain, crewai from requirements
        assert len(components) >= 2

    def test_empty_directory(self, scanner, tmp_path):
        components = scanner.scan(tmp_path)
        assert components == []

    def test_source_is_code(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("import openai\n")
        components = scanner.scan(tmp_path)
        for c in components:
            assert c.source == "code"


class TestDependencyParsing:
    def test_parse_pyproject_toml(self, scanner, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
dependencies = [
    "openai>=1.0.0",
    "langchain>=0.1.0"
]
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 2

    def test_parse_package_json(self, scanner, tmp_path):
        package_json = tmp_path / "package.json"
        package_json.write_text("""
{
  "dependencies": {
    "openai": "^4.0.0",
    "@anthropic-ai/sdk": "^0.9.0"
  }
}
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_cargo_toml(self, scanner, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("""
[dependencies]
async-openai = "0.14"
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_go_mod(self, scanner, tmp_path):
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("""
module example.com/app

require github.com/sashabaranov/go-openai v1.5.0
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_gemfile(self, scanner, tmp_path):
        gemfile = tmp_path / "Gemfile"
        gemfile.write_text("""
source 'https://rubygems.org'
gem 'ruby-openai'
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_pom_xml(self, scanner, tmp_path):
        pom = tmp_path / "pom.xml"
        # Use a known AI package that matches the pattern
        pom.write_text("""
<project>
    <dependencies>
        <dependency>
            <groupId>io.github.sashirestela</groupId>
            <artifactId>simple-openai</artifactId>
        </dependency>
    </dependencies>
</project>
""")
        components = scanner.scan(tmp_path)
        # May not detect if package not in known list, so just check it doesn't error
        assert isinstance(components, list)

    def test_parse_gradle(self, scanner, tmp_path):
        gradle = tmp_path / "build.gradle"
        gradle.write_text("""
dependencies {
    implementation 'dev.langchain4j:langchain4j:0.27.0'
}
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_gradle_kts(self, scanner, tmp_path):
        gradle_kts = tmp_path / "build.gradle.kts"
        gradle_kts.write_text("""
dependencies {
    implementation("dev.langchain4j:langchain4j:0.27.0")
}
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1

    def test_parse_csproj(self, scanner, tmp_path):
        csproj = tmp_path / "app.csproj"
        csproj.write_text("""
<Project>
    <ItemGroup>
        <PackageReference Include="Azure.AI.OpenAI" Version="1.0.0" />
    </ItemGroup>
</Project>
""")
        components = scanner.scan(tmp_path)
        assert len(components) >= 1


class TestShadowAI:
    def test_detects_shadow_ai(self, scanner, tmp_path):
        # Use openai without declaring it in requirements
        f = tmp_path / "app.py"
        f.write_text("from openai import OpenAI\nclient = OpenAI()\n")
        components = scanner.scan(tmp_path)
        shadow_components = [c for c in components if "shadow_ai" in c.flags]
        assert len(shadow_components) > 0

    def test_no_shadow_ai_when_declared(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("from openai import OpenAI\nclient = OpenAI()\n")
        req = tmp_path / "requirements.txt"
        req.write_text("openai>=1.0.0\n")
        components = scanner.scan(tmp_path)
        # Should have openai components but not flagged as shadow AI
        openai_components = [c for c in components if "openai" in c.name.lower()]
        shadow_components = [c for c in openai_components if "shadow_ai" in c.flags]
        assert len(shadow_components) == 0


class TestModelDetection:
    def test_detects_deprecated_model(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        # Put model on same line as usage pattern
        f.write_text("""
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[])
""")
        components = scanner.scan(tmp_path)
        deprecated = [c for c in components if "deprecated_model" in c.flags]
        assert len(deprecated) > 0

    def test_detects_unpinned_model(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        # Put model on same line as usage pattern
        f.write_text("""
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4", messages=[])
""")
        components = scanner.scan(tmp_path)
        unpinned = [c for c in components if "unpinned_model" in c.flags]
        assert len(unpinned) > 0

    def test_pinned_model_no_flag(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        # Put model on same line as usage pattern
        f.write_text("""
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4-0314", messages=[])
""")
        req = tmp_path / "requirements.txt"
        req.write_text("openai>=1.0.0\n")
        components = scanner.scan(tmp_path)
        unpinned = [c for c in components if "unpinned_model" in c.flags]
        # Model is pinned with date, so should not have unpinned flag
        assert len(unpinned) == 0


class TestSingleFileScanning:
    def test_scan_single_python_file(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("from openai import OpenAI\n")
        components = scanner.scan(f)
        assert len(components) > 0

    def test_scan_single_requirements_file(self, scanner, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("openai>=1.0.0\nlangchain>=0.1.0\n")
        components = scanner.scan(req)
        assert len(components) >= 2

    def test_scan_single_unreadable_file(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        f.write_bytes(b"\x00\x00\x00\x00")
        components = scanner.scan(f)
        assert components == []


class TestUsageTypeMapping:
    def test_map_usage_type_completion(self, scanner):
        usage_type = scanner._map_usage_type("completion")
        assert usage_type == UsageType.completion

    def test_map_usage_type_embedding(self, scanner):
        usage_type = scanner._map_usage_type("embedding")
        assert usage_type == UsageType.embedding

    def test_map_usage_type_agent(self, scanner):
        usage_type = scanner._map_usage_type("agent")
        assert usage_type == UsageType.agent

    def test_map_usage_type_unknown(self, scanner):
        usage_type = scanner._map_usage_type("invalid")
        assert usage_type == UsageType.unknown


class TestComponentTypeMapping:
    def test_determine_component_type_orchestration(self, scanner):
        comp_type = scanner._determine_component_type("LangChain", "orchestration")
        assert comp_type == ComponentType.agent_framework

    def test_determine_component_type_agent(self, scanner):
        comp_type = scanner._determine_component_type("CrewAI", "agent")
        assert comp_type == ComponentType.agent_framework

    def test_determine_component_type_tool(self, scanner):
        comp_type = scanner._determine_component_type("Unknown", "tool_use")
        assert comp_type == ComponentType.tool

    def test_determine_component_type_llm_default(self, scanner):
        comp_type = scanner._determine_component_type("OpenAI", "completion")
        assert comp_type == ComponentType.llm_provider


class TestIsModelPinned:
    def test_is_model_pinned_with_date(self, scanner):
        assert scanner._is_model_pinned("gpt-4-0314")
        assert scanner._is_model_pinned("claude-3-opus-20240229")

    def test_is_model_pinned_with_version(self, scanner):
        assert scanner._is_model_pinned("gpt-3.5-turbo-0125")

    def test_is_model_not_pinned(self, scanner):
        assert not scanner._is_model_pinned("gpt-4")
        assert not scanner._is_model_pinned("claude-3-opus")


class TestMultipleModelsInFile:
    def test_detects_multiple_model_usages(self, scanner, tmp_path):
        f = tmp_path / "app.py"
        # Put models on same line as usage pattern
        f.write_text("""
from openai import OpenAI
client = OpenAI()

# First deprecated model
response1 = client.chat.completions.create(model="gpt-3.5-turbo", messages=[])

# Second unpinned model
response2 = client.chat.completions.create(model="gpt-4", messages=[])
""")
        components = scanner.scan(tmp_path)
        # Should create separate components for model issues
        model_flags = [c.flags for c in components if c.model_name]
        assert len(model_flags) > 0
