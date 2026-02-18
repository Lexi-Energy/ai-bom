"""Tests for TruseraLLMInterceptor."""

from unittest.mock import Mock

import pytest

from trusera_sdk.exceptions import PolicyViolationError
from trusera_sdk.integrations.llm_interceptor import (
    TruseraLLMInterceptor,
    _extract_message_texts,
    _extract_tool_calls,
)


class TestLLMInterceptorInit:
    def test_default(self):
        i = TruseraLLMInterceptor()
        assert i._redactor is None

    def test_with_pii_redaction(self):
        i = TruseraLLMInterceptor(redact_pii=True)
        assert i._redactor is not None


class TestLLMInterceptorEvaluation:
    def test_no_cache_allows(self):
        i = TruseraLLMInterceptor()
        allowed, reason = i._evaluate("llm_call", "gpt-4")
        assert allowed is True

    def test_cache_deny(self, deny_all_cache):
        i = TruseraLLMInterceptor(policy_cache=deny_all_cache, enforcement="block")
        allowed, reason = i._evaluate("llm_call", "gpt-4")
        assert allowed is False

    def test_block_raises(self, deny_all_cache):
        i = TruseraLLMInterceptor(policy_cache=deny_all_cache, enforcement="block")
        with pytest.raises(PolicyViolationError):
            i._enforce(False, "denied", "llm_call", "gpt-4")


class TestLLMInterceptorWrap:
    def test_wrap_openai(self, allow_all_cache):
        i = TruseraLLMInterceptor(policy_cache=allow_all_cache)
        mock_client = Mock()
        mock_completions = Mock()
        original_create = Mock(return_value=Mock(choices=[]))
        mock_completions.create = original_create
        mock_client.chat.completions = mock_completions

        i.wrap_openai(mock_client)

        # The create method should now be wrapped
        assert mock_client.chat.completions.create is not original_create

    def test_wrap_anthropic(self, allow_all_cache):
        i = TruseraLLMInterceptor(policy_cache=allow_all_cache)
        mock_client = Mock()
        mock_messages = Mock()
        original_create = Mock(return_value=Mock(content=[]))
        mock_messages.create = original_create
        mock_client.messages = mock_messages

        i.wrap_anthropic(mock_client)

        assert mock_client.messages.create is not original_create


class TestExtractHelpers:
    def test_extract_message_texts(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _extract_message_texts(msgs)
        assert result == ["Hello", "Hi there"]

    def test_extract_message_texts_multipart(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "url": "..."},
                ],
            }
        ]
        result = _extract_message_texts(msgs)
        assert result == ["Hello"]

    def test_extract_message_texts_empty(self):
        assert _extract_message_texts([]) == []
        assert _extract_message_texts(None) == []

    def test_extract_tool_calls_openai(self):
        fn = Mock()
        fn.name = "search_web"
        tc = Mock()
        tc.function = fn

        msg = Mock()
        msg.tool_calls = [tc]

        choice = Mock()
        choice.message = msg

        response = Mock()
        response.choices = [choice]
        # No Anthropic content attribute
        del response.content

        result = _extract_tool_calls(response)
        assert result == ["search_web"]

    def test_extract_tool_calls_anthropic(self):
        block = Mock()
        block.type = "tool_use"
        block.name = "calculator"

        response = Mock()
        del response.choices  # not OpenAI
        response.content = [block]

        result = _extract_tool_calls(response)
        assert result == ["calculator"]

    def test_extract_tool_calls_none(self):
        response = Mock(spec=[])  # no choices, no content
        result = _extract_tool_calls(response)
        assert result == []
