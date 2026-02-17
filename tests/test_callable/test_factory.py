"""Tests for callable factory and adapter registry."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_bom.callable import (
    CallableResult,
    create_callable,
    get_callables,
    get_callables_from_cdx,
)
from ai_bom.callable.adapters import ADAPTERS, get_adapter_class
from ai_bom.models import (
    AIComponent,
    ComponentType,
    ScanResult,
    SourceLocation,
)


def _make_component(
    name: str = "openai",
    provider: str = "openai",
    model_name: str = "gpt-4o",
    ctype: ComponentType = ComponentType.llm_provider,
) -> AIComponent:
    return AIComponent(
        name=name,
        type=ctype,
        provider=provider,
        model_name=model_name,
        location=SourceLocation(file_path="main.py", line_number=10),
    )


# ── Adapter registry ──────────────────────────────────────────────────


class TestAdapterRegistry:
    def test_all_providers_registered(self) -> None:
        expected = {
            "openai", "anthropic", "google", "bedrock", "aws", "ollama", "mistral", "cohere",
        }
        assert set(ADAPTERS.keys()) == expected

    def test_get_adapter_class_openai(self) -> None:
        cls = get_adapter_class("openai")
        assert cls.__name__ == "OpenAIAdapter"

    def test_get_adapter_class_case_insensitive(self) -> None:
        cls = get_adapter_class("OpenAI")
        assert cls.__name__ == "OpenAIAdapter"

    def test_get_adapter_class_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unsupported provider"):
            get_adapter_class("unknown_provider")

    def test_aws_alias(self) -> None:
        cls = get_adapter_class("aws")
        assert cls.__name__ == "BedrockAdapter"


# ── create_callable ───────────────────────────────────────────────────


class TestCreateCallable:
    def test_creates_adapter(self) -> None:
        comp = _make_component()
        adapter = create_callable(comp)
        assert adapter.model_name == "gpt-4o"
        assert adapter.provider == "openai"

    def test_uses_component_name_when_no_model_name(self) -> None:
        comp = _make_component(model_name="")
        adapter = create_callable(comp)
        assert adapter.model_name == "openai"

    def test_forwards_kwargs(self) -> None:
        comp = _make_component()
        adapter = create_callable(comp, api_key="sk-test")
        assert adapter._kwargs["api_key"] == "sk-test"

    def test_raises_on_empty_provider(self) -> None:
        comp = _make_component(provider="")
        with pytest.raises(ValueError, match="no provider set"):
            create_callable(comp)

    def test_raises_on_unknown_provider(self) -> None:
        comp = _make_component(provider="unknown")
        with pytest.raises(KeyError):
            create_callable(comp)


# ── get_callables ─────────────────────────────────────────────────────


class TestGetCallables:
    def test_from_scan_result(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(provider="openai"),
                _make_component(
                    name="langchain",
                    provider="",
                    model_name="",
                    ctype=ComponentType.agent_framework,
                ),
                _make_component(
                    name="claude-3",
                    provider="anthropic",
                    model_name="claude-3-5-sonnet-20241022",
                ),
            ],
        )
        callables = get_callables(result)
        assert len(callables) == 2
        providers = {c.provider for c in callables}
        assert providers == {"openai", "anthropic"}

    def test_from_component_list(self) -> None:
        components = [_make_component()]
        callables = get_callables(components)
        assert len(callables) == 1

    def test_skips_non_model_types(self) -> None:
        comp = _make_component(ctype=ComponentType.tool)
        callables = get_callables([comp])
        assert len(callables) == 0

    def test_skips_unsupported_providers(self) -> None:
        comp = _make_component(provider="some_unknown")
        callables = get_callables([comp])
        assert len(callables) == 0

    def test_empty_input(self) -> None:
        assert get_callables([]) == []
        assert get_callables(ScanResult(target_path=".")) == []


# ── get_callables_from_cdx ────────────────────────────────────────────


class TestGetCallablesFromCdx:
    def _sample_cdx(self) -> dict[str, Any]:
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "components": [
                {
                    "bom-ref": "abc-123",
                    "type": "machine-learning-model",
                    "name": "openai",
                    "properties": [
                        {"name": "trusera:provider", "value": "openai"},
                        {"name": "trusera:model_name", "value": "gpt-4o"},
                        {"name": "trusera:risk_score", "value": "45"},
                    ],
                },
                {
                    "bom-ref": "def-456",
                    "type": "framework",
                    "name": "langchain",
                    "properties": [
                        {"name": "trusera:provider", "value": ""},
                        {"name": "trusera:risk_score", "value": "20"},
                    ],
                },
                {
                    "bom-ref": "ghi-789",
                    "type": "container",
                    "name": "ollama",
                    "properties": [],
                },
            ],
        }

    def test_parses_cdx_components(self) -> None:
        callables = get_callables_from_cdx(self._sample_cdx())
        assert len(callables) == 1
        assert callables[0].provider == "openai"
        assert callables[0].model_name == "gpt-4o"

    def test_empty_cdx(self) -> None:
        assert get_callables_from_cdx({}) == []
        assert get_callables_from_cdx({"components": []}) == []

    def test_falls_back_to_component_name(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "mistral-large-latest",
                    "properties": [
                        {"name": "trusera:provider", "value": "mistral"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].model_name == "mistral-large-latest"


# ── Adapter __call__ (mocked SDKs) ───────────────────────────────────


class TestOpenAIAdapterCall:
    def test_call(self) -> None:
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from ai_bom.callable.adapters.openai import OpenAIAdapter

            adapter = OpenAIAdapter(model_name="gpt-4o", provider="openai", api_key="sk-test")
            result = adapter("Tell me a joke")

        assert isinstance(result, CallableResult)
        assert result.text == "Hello!"
        assert result.usage["total_tokens"] == 15
        mock_client.chat.completions.create.assert_called_once()


class TestAnthropicAdapterCall:
    def test_call(self) -> None:
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hi there"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 8
        mock_usage.output_tokens = 3

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = mock_usage
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from ai_bom.callable.adapters.anthropic import AnthropicAdapter

            adapter = AnthropicAdapter(
                model_name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                api_key="sk-ant-test",
            )
            result = adapter("Hello")

        assert isinstance(result, CallableResult)
        assert result.text == "Hi there"
        assert result.usage["input_tokens"] == 8


class TestBedrockAdapterCall:
    def test_call(self) -> None:
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Bedrock says hi"}],
                },
            },
            "usage": {
                "inputTokens": 12,
                "outputTokens": 4,
                "totalTokens": 16,
            },
        }

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
            )
            result = adapter("Hello Bedrock")

        assert isinstance(result, CallableResult)
        assert result.text == "Bedrock says hi"
        assert result.usage["total_tokens"] == 16


class TestOllamaAdapterCall:
    def test_call(self) -> None:
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        mock_client.chat.return_value = {
            "message": {"content": "Local model response"},
            "prompt_eval_count": 20,
            "eval_count": 10,
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(model_name="llama3", provider="ollama")
            result = adapter("Hello local")

        assert isinstance(result, CallableResult)
        assert result.text == "Local model response"
        assert result.usage["prompt_tokens"] == 20


class TestMistralAdapterCall:
    def test_call(self) -> None:
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.Mistral.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Mistral response"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 6
        mock_usage.completion_tokens = 4
        mock_usage.total_tokens = 10

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "mistral-large-latest"
        mock_client.chat.complete.return_value = mock_response

        with patch.dict(sys.modules, {"mistralai": mock_mistralai}):
            from ai_bom.callable.adapters.mistral import MistralAdapter

            adapter = MistralAdapter(model_name="mistral-large-latest", provider="mistral")
            result = adapter("Hi Mistral")

        assert isinstance(result, CallableResult)
        assert result.text == "Mistral response"


class TestCohereAdapterCall:
    def test_call(self) -> None:
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = "Cohere response"

        mock_tokens = MagicMock()
        mock_tokens.input_tokens = 7
        mock_tokens.output_tokens = 3

        mock_usage = MagicMock()
        mock_usage.tokens = mock_tokens

        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage
        mock_client.chat.return_value = mock_response

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(model_name="command-r-plus", provider="cohere")
            result = adapter("Hi Cohere")

        assert isinstance(result, CallableResult)
        assert result.text == "Cohere response"


# ── SDK check ─────────────────────────────────────────────────────────


class TestSdkCheck:
    def test_import_error_message(self) -> None:
        from ai_bom.callable.adapters._base import BaseAdapter

        class _TestAdapter(BaseAdapter):
            SDK_PACKAGE = "nonexistent_sdk_xyz"

            def _get_client(self) -> Any:
                self._check_sdk()

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                return CallableResult(text="")

        adapter = _TestAdapter(model_name="m", provider="test")
        with pytest.raises(ImportError, match="nonexistent_sdk_xyz"):
            adapter._get_client()


# ── Google adapter ────────────────────────────────────────────────────


class TestGoogleAdapterCall:
    def _make_mock_genai(self) -> MagicMock:
        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_meta = MagicMock()
        mock_meta.prompt_token_count = 12
        mock_meta.candidates_token_count = 8
        mock_meta.total_token_count = 20

        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        mock_response.usage_metadata = mock_meta
        mock_model.generate_content.return_value = mock_response
        return mock_genai

    def test_call_basic(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(
                model_name="gemini-1.5-pro",
                provider="google",
                api_key="AIza-test",
            )
            result = adapter("Hello Gemini")

        assert isinstance(result, CallableResult)
        assert result.text == "Gemini response"
        assert result.provider == "google"
        assert result.model_name == "gemini-1.5-pro"
        assert result.usage["total_tokens"] == 20
        assert result.usage["prompt_tokens"] == 12
        assert result.usage["completion_tokens"] == 8

    def test_call_usage_metadata_sets_token_counts(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(model_name="gemini-pro", provider="google")
            result = adapter("Test prompt")

        assert result.usage["prompt_tokens"] == 12
        assert result.usage["completion_tokens"] == 8

    def test_call_no_usage_metadata(self) -> None:
        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_response = MagicMock(spec=["text"])
        mock_response.text = "No usage"
        mock_model.generate_content.return_value = mock_response

        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(model_name="gemini-pro", provider="google")
            result = adapter("Test")

        assert result.text == "No usage"
        assert result.usage == {}

    def test_call_with_temperature_and_max_tokens(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_model = mock_genai.GenerativeModel.return_value
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(model_name="gemini-pro", provider="google")
            adapter("Prompt", temperature=0.7, max_tokens=256)

        call_kwargs = mock_model.generate_content.call_args[1]
        cfg = call_kwargs["generation_config"]
        assert cfg["temperature"] == 0.7
        assert cfg["max_output_tokens"] == 256

    def test_call_without_kwargs_passes_none_config(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_model = mock_genai.GenerativeModel.return_value
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(model_name="gemini-pro", provider="google")
            adapter("Prompt")

        call_kwargs = mock_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"] is None

    def test_configure_called_with_api_key(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(
                model_name="gemini-pro", provider="google", api_key="key-xyz"
            )
            adapter("Prompt")

        mock_genai.configure.assert_called_once_with(api_key="key-xyz")

    def test_client_cached_across_calls(self) -> None:
        mock_genai = self._make_mock_genai()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {"google": mock_google, "google.generativeai": mock_genai},
        ):
            from ai_bom.callable.adapters.google import GoogleAdapter

            adapter = GoogleAdapter(model_name="gemini-pro", provider="google")
            adapter("First call")
            adapter("Second call")

        # GenerativeModel should only be constructed once
        assert mock_genai.GenerativeModel.call_count == 1


# ── Bedrock adapter branches ──────────────────────────────────────────


class TestBedrockAdapterBranches:
    def _make_mock_boto3(self, text: str = "Bedrock result") -> MagicMock:
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": text}],
                },
            },
            "usage": {
                "inputTokens": 5,
                "outputTokens": 3,
                "totalTokens": 8,
            },
        }
        return mock_boto3

    def test_inference_config_with_temperature(self) -> None:
        mock_boto3 = self._make_mock_boto3()
        mock_client = mock_boto3.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
            )
            result = adapter("Hello", temperature=0.5)

        assert isinstance(result, CallableResult)
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.5
        assert "maxTokens" not in call_kwargs["inferenceConfig"]

    def test_inference_config_with_max_tokens(self) -> None:
        mock_boto3 = self._make_mock_boto3()
        mock_client = mock_boto3.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
            )
            result = adapter("Hello", max_tokens=512)

        assert isinstance(result, CallableResult)
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 512
        assert "temperature" not in call_kwargs["inferenceConfig"]

    def test_inference_config_with_both_params(self) -> None:
        mock_boto3 = self._make_mock_boto3()
        mock_client = mock_boto3.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
            )
            adapter("Hello", temperature=0.3, max_tokens=100)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"] == {"temperature": 0.3, "maxTokens": 100}

    def test_no_inference_config_when_no_params(self) -> None:
        mock_boto3 = self._make_mock_boto3()
        mock_client = mock_boto3.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
            )
            adapter("Hello")

        call_kwargs = mock_client.converse.call_args[1]
        assert "inferenceConfig" not in call_kwargs

    def test_params_from_constructor_kwargs(self) -> None:
        mock_boto3 = self._make_mock_boto3()
        mock_client = mock_boto3.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from ai_bom.callable.adapters.bedrock import BedrockAdapter

            adapter = BedrockAdapter(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
                temperature=0.9,
            )
            adapter("Hello")

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.9


# ── Cohere adapter branches ───────────────────────────────────────────


class TestCohereAdapterBranches:
    def _make_mock_cohere(self, text: str = "Cohere reply") -> MagicMock:
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = text
        mock_tokens = MagicMock()
        mock_tokens.input_tokens = 4
        mock_tokens.output_tokens = 2
        mock_usage = MagicMock()
        mock_usage.tokens = mock_tokens
        mock_message = MagicMock()
        mock_message.content = [mock_block]
        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage
        mock_client.chat.return_value = mock_response
        return mock_cohere

    def test_temperature_forwarded(self) -> None:
        mock_cohere = self._make_mock_cohere()
        mock_client = mock_cohere.ClientV2.return_value

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(model_name="command-r-plus", provider="cohere")
            result = adapter("Hello", temperature=0.8)

        assert isinstance(result, CallableResult)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.8
        assert "max_tokens" not in call_kwargs

    def test_max_tokens_forwarded(self) -> None:
        mock_cohere = self._make_mock_cohere()
        mock_client = mock_cohere.ClientV2.return_value

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(model_name="command-r-plus", provider="cohere")
            result = adapter("Hello", max_tokens=200)

        assert isinstance(result, CallableResult)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["max_tokens"] == 200
        assert "temperature" not in call_kwargs

    def test_both_params_forwarded(self) -> None:
        mock_cohere = self._make_mock_cohere()
        mock_client = mock_cohere.ClientV2.return_value

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(model_name="command-r-plus", provider="cohere")
            adapter("Hello", temperature=0.4, max_tokens=150)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.4
        assert call_kwargs["max_tokens"] == 150

    def test_no_optional_params_when_absent(self) -> None:
        mock_cohere = self._make_mock_cohere()
        mock_client = mock_cohere.ClientV2.return_value

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(model_name="command-r-plus", provider="cohere")
            adapter("Hello")

        call_kwargs = mock_client.chat.call_args[1]
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs

    def test_params_from_constructor_kwargs(self) -> None:
        mock_cohere = self._make_mock_cohere()
        mock_client = mock_cohere.ClientV2.return_value

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            from ai_bom.callable.adapters.cohere import CohereAdapter

            adapter = CohereAdapter(
                model_name="command-r-plus",
                provider="cohere",
                temperature=0.2,
                max_tokens=50,
            )
            adapter("Hello")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 50


# ── Ollama adapter branches ───────────────────────────────────────────


class TestOllamaAdapterBranches:
    def _make_mock_ollama(self, text: str = "Ollama reply") -> MagicMock:
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": text},
            "prompt_eval_count": 7,
            "eval_count": 4,
        }
        return mock_ollama

    def test_host_kwarg_used_when_provided(self) -> None:
        mock_ollama = self._make_mock_ollama()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(
                model_name="llama3", provider="ollama", host="http://localhost:11434"
            )
            adapter("Hello")

        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")

    def test_no_host_kwarg_uses_default_client(self) -> None:
        mock_ollama = self._make_mock_ollama()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(model_name="llama3", provider="ollama")
            adapter("Hello")

        mock_ollama.Client.assert_called_once_with()

    def test_temperature_forwarded_in_options(self) -> None:
        mock_ollama = self._make_mock_ollama()
        mock_client = mock_ollama.Client.return_value

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(model_name="llama3", provider="ollama")
            result = adapter("Hello", temperature=0.6)

        assert isinstance(result, CallableResult)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"] == {"temperature": 0.6}

    def test_no_temperature_passes_none_options(self) -> None:
        mock_ollama = self._make_mock_ollama()
        mock_client = mock_ollama.Client.return_value

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(model_name="llama3", provider="ollama")
            adapter("Hello")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"] is None

    def test_temperature_from_constructor_kwargs(self) -> None:
        mock_ollama = self._make_mock_ollama()
        mock_client = mock_ollama.Client.return_value

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from ai_bom.callable.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(
                model_name="llama3", provider="ollama", temperature=0.1
            )
            adapter("Hello")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"] == {"temperature": 0.1}


# ── get_callables_from_cdx edge cases ────────────────────────────────


class TestGetCallablesFromCdxEdgeCases:
    def test_unsupported_type_skipped(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "container",
                    "name": "my-container",
                    "properties": [
                        {"name": "trusera:provider", "value": "openai"},
                        {"name": "trusera:model_name", "value": "gpt-4o"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert callables == []

    def test_missing_properties_array_skipped(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "some-model",
                    # no "properties" key at all
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert callables == []

    def test_unsupported_provider_skipped(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "exotic-model",
                    "properties": [
                        {"name": "trusera:provider", "value": "exotic_ai_corp"},
                        {"name": "trusera:model_name", "value": "exotic-v1"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert callables == []

    def test_empty_provider_skipped(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "unnamed",
                    "properties": [
                        {"name": "trusera:provider", "value": ""},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert callables == []

    def test_framework_type_processed(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "framework",
                    "name": "custom-wrapper",
                    "properties": [
                        {"name": "trusera:provider", "value": "anthropic"},
                        {"name": "trusera:model_name", "value": "claude-3-haiku-20240307"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].provider == "anthropic"

    def test_library_type_processed(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "library",
                    "name": "my-lib",
                    "properties": [
                        {"name": "trusera:provider", "value": "mistral"},
                        {"name": "trusera:model_name", "value": "mistral-small-latest"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].model_name == "mistral-small-latest"

    def test_mixed_valid_and_invalid_components(self) -> None:
        cdx: dict[str, Any] = {
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "openai",
                    "properties": [
                        {"name": "trusera:provider", "value": "openai"},
                        {"name": "trusera:model_name", "value": "gpt-4o-mini"},
                    ],
                },
                {
                    "type": "container",
                    "name": "postgres",
                    "properties": [
                        {"name": "trusera:provider", "value": "openai"},
                    ],
                },
                {
                    "type": "machine-learning-model",
                    "name": "unknown-model",
                    "properties": [
                        {"name": "trusera:provider", "value": "nonexistent_provider"},
                    ],
                },
            ],
        }
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].model_name == "gpt-4o-mini"


# ── get_callables edge cases ──────────────────────────────────────────


class TestGetCallablesEdgeCases:
    def test_model_type_included(self) -> None:
        comp = _make_component(
            name="my-model",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            ctype=ComponentType.model,
        )
        callables = get_callables([comp])
        assert len(callables) == 1
        assert callables[0].provider == "anthropic"

    def test_multiple_components_same_provider(self) -> None:
        comp1 = _make_component(
            name="openai-1",
            provider="openai",
            model_name="gpt-4o",
        )
        comp2 = _make_component(
            name="openai-2",
            provider="openai",
            model_name="gpt-4o-mini",
        )
        callables = get_callables([comp1, comp2])
        assert len(callables) == 2
        model_names = {c.model_name for c in callables}
        assert model_names == {"gpt-4o", "gpt-4o-mini"}

    def test_tool_type_skipped(self) -> None:
        comp = _make_component(ctype=ComponentType.tool)
        callables = get_callables([comp])
        assert callables == []

    def test_agent_framework_type_skipped(self) -> None:
        comp = _make_component(
            name="langchain",
            provider="",
            model_name="",
            ctype=ComponentType.agent_framework,
        )
        callables = get_callables([comp])
        assert callables == []

    def test_scan_result_with_model_type(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(
                    name="cohere-embed",
                    provider="cohere",
                    model_name="embed-english-v3",
                    ctype=ComponentType.model,
                ),
            ],
        )
        callables = get_callables(result)
        assert len(callables) == 1
        assert callables[0].provider == "cohere"

    def test_model_type_with_empty_provider_skipped(self) -> None:
        # Hits the `if not component.provider: continue` guard on a callable type
        comp = _make_component(
            name="unnamed-model",
            provider="",
            model_name="",
            ctype=ComponentType.model,
        )
        callables = get_callables([comp])
        assert callables == []


# ── BaseAdapter empty SDK_PACKAGE ─────────────────────────────────────


class TestBaseAdapterEmptySdkPackage:
    def test_check_sdk_noop_when_empty(self) -> None:
        from ai_bom.callable.adapters._base import BaseAdapter

        class _NoSdkAdapter(BaseAdapter):
            SDK_PACKAGE = ""

            def _get_client(self) -> Any:
                self._check_sdk()
                return None

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                self._get_client()
                return CallableResult(text="ok")

        adapter = _NoSdkAdapter(model_name="m", provider="test")
        # Must not raise even though SDK_PACKAGE is empty
        result = adapter("Hello")
        assert result.text == "ok"

    def test_check_sdk_called_multiple_times_does_not_raise(self) -> None:
        from ai_bom.callable.adapters._base import BaseAdapter

        class _RepeatedCheckAdapter(BaseAdapter):
            SDK_PACKAGE = ""

            def _get_client(self) -> Any:
                self._check_sdk()
                self._check_sdk()
                return None

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                return CallableResult(text="repeated")

        adapter = _RepeatedCheckAdapter(model_name="m", provider="test")
        # Calling _get_client twice should never raise
        adapter._get_client()
        adapter._get_client()
        result = adapter("test")
        assert result.text == "repeated"


# ── CallableModel protocol edge cases ─────────────────────────────────


class TestCallableModelProtocolEdgeCases:
    def test_object_without_model_name_fails_protocol(self) -> None:
        from ai_bom.callable._protocol import CallableModel

        class _NoModelName:
            provider: str = "openai"

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                return CallableResult(text="")

        obj = _NoModelName()
        assert not isinstance(obj, CallableModel)

    def test_object_without_provider_fails_protocol(self) -> None:
        from ai_bom.callable._protocol import CallableModel

        class _NoProvider:
            model_name: str = "gpt-4o"

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                return CallableResult(text="")

        obj = _NoProvider()
        assert not isinstance(obj, CallableModel)

    def test_object_without_call_fails_protocol(self) -> None:
        from ai_bom.callable._protocol import CallableModel

        class _NoCall:
            model_name: str = "gpt-4o"
            provider: str = "openai"

        obj = _NoCall()
        assert not isinstance(obj, CallableModel)

    def test_valid_object_satisfies_protocol(self) -> None:
        from ai_bom.callable._protocol import CallableModel

        class _ValidModel:
            model_name: str = "gpt-4o"
            provider: str = "openai"

            def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
                return CallableResult(text="ok")

        obj = _ValidModel()
        assert isinstance(obj, CallableModel)

    def test_base_adapter_satisfies_protocol(self) -> None:
        from ai_bom.callable._protocol import CallableModel

        # OpenAI adapter (no SDK call needed for isinstance check)
        comp = _make_component(provider="openai", model_name="gpt-4o")
        adapter = create_callable(comp)
        assert isinstance(adapter, CallableModel)


# ── Roundtrip: ScanResult → CycloneDX → get_callables_from_cdx ───────


class TestCreateCallableFromRealCycloneDX:
    def test_roundtrip_single_component(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(
                    name="openai",
                    provider="openai",
                    model_name="gpt-4o",
                    ctype=ComponentType.llm_provider,
                ),
            ],
        )
        cdx = result.to_cyclonedx()
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].provider == "openai"
        assert callables[0].model_name == "gpt-4o"

    def test_roundtrip_filters_non_callable_types(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(
                    name="openai",
                    provider="openai",
                    model_name="gpt-4o",
                    ctype=ComponentType.llm_provider,
                ),
                _make_component(
                    name="langchain",
                    provider="",
                    model_name="",
                    ctype=ComponentType.agent_framework,
                ),
                _make_component(
                    name="my-tool",
                    provider="",
                    model_name="",
                    ctype=ComponentType.tool,
                ),
            ],
        )
        cdx = result.to_cyclonedx()
        callables = get_callables_from_cdx(cdx)
        # Only openai should be callable; agent_framework maps to "framework"
        # but has no provider; tool maps to "library" but has no provider
        assert len(callables) == 1
        assert callables[0].provider == "openai"

    def test_roundtrip_multiple_providers(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(
                    name="openai",
                    provider="openai",
                    model_name="gpt-4o",
                    ctype=ComponentType.llm_provider,
                ),
                _make_component(
                    name="anthropic",
                    provider="anthropic",
                    model_name="claude-3-5-sonnet-20241022",
                    ctype=ComponentType.llm_provider,
                ),
            ],
        )
        cdx = result.to_cyclonedx()
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 2
        providers = {c.provider for c in callables}
        assert providers == {"openai", "anthropic"}

    def test_roundtrip_model_name_fallback_to_component_name(self) -> None:
        result = ScanResult(
            target_path=".",
            components=[
                _make_component(
                    name="mistral-large-latest",
                    provider="mistral",
                    model_name="",
                    ctype=ComponentType.llm_provider,
                ),
            ],
        )
        cdx = result.to_cyclonedx()
        callables = get_callables_from_cdx(cdx)
        assert len(callables) == 1
        assert callables[0].model_name == "mistral-large-latest"


# ── Adapter repr / construction properties ────────────────────────────


class TestAdapterRepr:
    def test_openai_adapter_properties(self) -> None:
        comp = _make_component(provider="openai", model_name="gpt-4o-mini")
        adapter = create_callable(comp)
        assert adapter.model_name == "gpt-4o-mini"
        assert adapter.provider == "openai"
        assert adapter.__class__.__name__ == "OpenAIAdapter"

    def test_anthropic_adapter_properties(self) -> None:
        comp = _make_component(
            provider="anthropic", model_name="claude-3-5-haiku-20241022"
        )
        adapter = create_callable(comp)
        assert adapter.model_name == "claude-3-5-haiku-20241022"
        assert adapter.provider == "anthropic"
        assert adapter.__class__.__name__ == "AnthropicAdapter"

    def test_google_adapter_properties(self) -> None:
        comp = _make_component(provider="google", model_name="gemini-1.5-flash")
        adapter = create_callable(comp)
        assert adapter.model_name == "gemini-1.5-flash"
        assert adapter.provider == "google"
        assert adapter.__class__.__name__ == "GoogleAdapter"

    def test_bedrock_adapter_properties(self) -> None:
        comp = _make_component(
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )
        adapter = create_callable(comp)
        assert adapter.model_name == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert adapter.provider == "bedrock"
        assert adapter.__class__.__name__ == "BedrockAdapter"

    def test_ollama_adapter_properties(self) -> None:
        comp = _make_component(provider="ollama", model_name="llama3.1")
        adapter = create_callable(comp)
        assert adapter.model_name == "llama3.1"
        assert adapter.provider == "ollama"
        assert adapter.__class__.__name__ == "OllamaAdapter"

    def test_mistral_adapter_properties(self) -> None:
        comp = _make_component(provider="mistral", model_name="mistral-large-latest")
        adapter = create_callable(comp)
        assert adapter.model_name == "mistral-large-latest"
        assert adapter.provider == "mistral"
        assert adapter.__class__.__name__ == "MistralAdapter"

    def test_cohere_adapter_properties(self) -> None:
        comp = _make_component(provider="cohere", model_name="command-r-plus")
        adapter = create_callable(comp)
        assert adapter.model_name == "command-r-plus"
        assert adapter.provider == "cohere"
        assert adapter.__class__.__name__ == "CohereAdapter"

    def test_aws_alias_gives_bedrock_adapter(self) -> None:
        comp = _make_component(
            provider="aws",
            model_name="amazon.titan-text-express-v1",
        )
        adapter = create_callable(comp)
        assert adapter.__class__.__name__ == "BedrockAdapter"
        assert adapter.provider == "aws"

    def test_kwargs_stored_on_adapter(self) -> None:
        comp = _make_component(provider="openai", model_name="gpt-4o")
        adapter = create_callable(comp, api_key="sk-test-key", temperature=0.7)
        assert adapter._kwargs["api_key"] == "sk-test-key"
        assert adapter._kwargs["temperature"] == 0.7

    def test_client_initially_none(self) -> None:
        comp = _make_component(provider="openai", model_name="gpt-4o")
        adapter = create_callable(comp)
        assert adapter._client is None
