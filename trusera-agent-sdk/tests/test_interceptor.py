"""Tests for TruseraInterceptor."""

from unittest.mock import Mock

import httpx
import pytest

from trusera_sdk.enforcement import EnforcementMode
from trusera_sdk.exceptions import PolicyViolationError
from trusera_sdk.interceptor import TruseraInterceptor


def _mock_response(status_code=200):
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    return resp


class TestInterceptorInit:
    def test_default_enforcement(self):
        i = TruseraInterceptor()
        assert i.enforcement == EnforcementMode.LOG
        assert not i._installed

    def test_string_enforcement(self):
        i = TruseraInterceptor(enforcement="block")
        assert i.enforcement == EnforcementMode.BLOCK

    def test_invalid_enforcement(self):
        with pytest.raises(ValueError, match="Invalid enforcement mode"):
            TruseraInterceptor(enforcement="bad")

    def test_client_base_url_excluded(self):
        client = Mock()
        client.base_url = "https://api.trusera.dev"
        i = TruseraInterceptor(client=client)
        assert not i._should_intercept("https://api.trusera.dev/v1/policies")

    def test_custom_excludes(self):
        i = TruseraInterceptor(exclude_patterns=[r"localhost:\d+"])
        assert not i._should_intercept("http://localhost:8080/health")
        assert i._should_intercept("https://example.com/api")


class TestInterceptorInstallUninstall:
    def test_install_uninstall(self):
        i = TruseraInterceptor()
        i.install()
        assert i._installed
        i.uninstall()
        assert not i._installed

    def test_install_twice_raises(self):
        i = TruseraInterceptor()
        i.install()
        with pytest.raises(RuntimeError, match="already installed"):
            i.install()
        i.uninstall()

    def test_uninstall_not_installed_raises(self):
        i = TruseraInterceptor()
        with pytest.raises(RuntimeError, match="not installed"):
            i.uninstall()

    def test_context_manager(self):
        with TruseraInterceptor() as i:
            assert i._installed
        assert not i._installed


class TestInterceptorEvaluation:
    def test_no_cache_allows(self):
        i = TruseraInterceptor()
        allowed, reason, pid = i._evaluate("https://example.com", "GET")
        assert allowed is True

    def test_cache_allow(self, allow_all_cache):
        i = TruseraInterceptor(policy_cache=allow_all_cache)
        allowed, reason, pid = i._evaluate("https://example.com", "GET")
        assert allowed is True

    def test_cache_deny(self, deny_all_cache):
        i = TruseraInterceptor(policy_cache=deny_all_cache)
        allowed, reason, pid = i._evaluate("https://evil.com", "GET")
        assert allowed is False


class TestInterceptorEnforcement:
    def test_block_mode_raises(self, deny_all_cache):
        i = TruseraInterceptor(enforcement="block", policy_cache=deny_all_cache)
        with pytest.raises(PolicyViolationError, match="http"):
            i._enforce(False, "denied", "GET", "https://evil.com", None)

    def test_warn_mode_allows(self, deny_all_cache):
        i = TruseraInterceptor(enforcement="warn", policy_cache=deny_all_cache)
        # Should not raise
        i._enforce(False, "denied", "GET", "https://evil.com", None)

    def test_log_mode_allows(self, deny_all_cache):
        i = TruseraInterceptor(enforcement="log", policy_cache=deny_all_cache)
        # Should not raise
        i._enforce(False, "denied", "GET", "https://evil.com", None)

    def test_allow_never_raises(self, allow_all_cache):
        i = TruseraInterceptor(enforcement="block", policy_cache=allow_all_cache)
        i._enforce(True, "allowed", "GET", "https://example.com", None)


class TestInterceptorHttpxSync:
    def test_intercept_allowed(self, allow_all_cache):
        i = TruseraInterceptor(policy_cache=allow_all_cache, enforcement="block")
        orig = Mock(return_value=_mock_response(200))
        client_self = Mock()
        request = Mock()
        request.url = "https://example.com/api"
        request.method = "GET"
        request.headers = {}

        resp = i._intercept_httpx_sync(orig, client_self, request)
        assert resp.status_code == 200
        orig.assert_called_once()

    def test_intercept_blocked(self, deny_all_cache):
        i = TruseraInterceptor(policy_cache=deny_all_cache, enforcement="block")
        orig = Mock(return_value=_mock_response(200))
        client_self = Mock()
        request = Mock()
        request.url = "https://evil.com/api"
        request.method = "GET"
        request.headers = {}

        with pytest.raises(PolicyViolationError):
            i._intercept_httpx_sync(orig, client_self, request)
        orig.assert_not_called()

    def test_intercept_excluded(self, deny_all_cache):
        i = TruseraInterceptor(
            policy_cache=deny_all_cache,
            enforcement="block",
            exclude_patterns=[r"example\.com"],
        )
        orig = Mock(return_value=_mock_response(200))
        client_self = Mock()
        request = Mock()
        request.url = "https://example.com/api"
        request.method = "GET"

        resp = i._intercept_httpx_sync(orig, client_self, request)
        assert resp.status_code == 200


class TestInterceptorRepr:
    def test_repr(self):
        i = TruseraInterceptor(enforcement="warn")
        s = repr(i)
        assert "TruseraInterceptor" in s
        assert "warn" in s
