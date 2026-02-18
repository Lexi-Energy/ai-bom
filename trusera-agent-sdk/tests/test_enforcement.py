"""Tests for EnforcementMode."""

import pytest

from trusera_sdk.enforcement import EnforcementMode


def test_enforcement_values():
    assert EnforcementMode.BLOCK.value == "block"
    assert EnforcementMode.WARN.value == "warn"
    assert EnforcementMode.LOG.value == "log"


def test_from_string_valid():
    assert EnforcementMode.from_string("block") == EnforcementMode.BLOCK
    assert EnforcementMode.from_string("WARN") == EnforcementMode.WARN
    assert EnforcementMode.from_string("Log") == EnforcementMode.LOG


def test_from_string_invalid():
    with pytest.raises(ValueError, match="Invalid enforcement mode"):
        EnforcementMode.from_string("invalid")


def test_enforcement_is_string_enum():
    assert isinstance(EnforcementMode.BLOCK, str)
    assert EnforcementMode.BLOCK == "block"
