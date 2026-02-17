"""Tests for callable protocol and result model."""

from __future__ import annotations

from typing import Any

from ai_bom.callable import CallableModel, CallableResult


class _FakeCallable:
    """Minimal implementation satisfying the CallableModel protocol."""

    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.provider = "fake"

    def __call__(self, prompt: str, **kwargs: Any) -> CallableResult:
        return CallableResult(
            text=f"echo: {prompt}", model_name=self.model_name, provider=self.provider,
        )


class TestCallableResult:
    def test_defaults(self) -> None:
        r = CallableResult(text="hello")
        assert r.text == "hello"
        assert r.model_name == ""
        assert r.provider == ""
        assert r.usage == {}
        assert r.raw is None

    def test_full_fields(self) -> None:
        raw_obj = {"id": "123"}
        r = CallableResult(
            text="response",
            model_name="gpt-4o",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw=raw_obj,
        )
        assert r.text == "response"
        assert r.model_name == "gpt-4o"
        assert r.provider == "openai"
        assert r.usage["prompt_tokens"] == 10
        assert r.raw is raw_obj

    def test_serialization(self) -> None:
        r = CallableResult(text="test", model_name="m", provider="p")
        d = r.model_dump()
        assert d["text"] == "test"
        assert d["model_name"] == "m"

        r2 = CallableResult.model_validate(d)
        assert r2.text == r.text


class TestCallableModelProtocol:
    def test_isinstance_check(self) -> None:
        obj = _FakeCallable()
        assert isinstance(obj, CallableModel)

    def test_fake_callable_works(self) -> None:
        obj = _FakeCallable()
        result = obj("hello world")
        assert result.text == "echo: hello world"
        assert result.model_name == "fake-model"

    def test_non_callable_fails(self) -> None:
        assert not isinstance("a string", CallableModel)
        assert not isinstance(42, CallableModel)
