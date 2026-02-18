"""Tests for PIIRedactor."""

from trusera_sdk.pii import PIIRedactor


class TestPIIRedactor:
    def setup_method(self):
        self.redactor = PIIRedactor()

    def test_redact_email(self):
        text = "Contact john.doe@example.com for details"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_redact_phone(self):
        text = "Call me at 555-123-4567"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_PHONE]" in result
        assert "555-123-4567" not in result

    def test_redact_phone_with_parens(self):
        text = "Phone: (555) 123-4567"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_PHONE]" in result

    def test_redact_ssn(self):
        text = "SSN: 123-45-6789"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_SSN]" in result
        assert "123-45-6789" not in result

    def test_redact_credit_card(self):
        text = "Card: 4111 1111 1111 1111"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_CREDIT_CARD]" in result
        assert "4111" not in result

    def test_redact_ip(self):
        text = "Server at 192.168.1.100"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_IPV4]" in result
        assert "192.168.1.100" not in result

    def test_redact_no_pii(self):
        text = "Hello world, nothing sensitive here"
        result = self.redactor.redact_text(text)
        assert result == text

    def test_redact_dict(self):
        data = {"email": "user@test.com", "name": "Alice", "count": 42}
        result = self.redactor.redact(data)
        assert "[REDACTED_EMAIL]" in result["email"]
        assert result["name"] == "Alice"
        assert result["count"] == 42

    def test_redact_nested_dict(self):
        data = {"user": {"contact": "user@test.com", "phone": "555-111-2222"}}
        result = self.redactor.redact(data)
        assert "[REDACTED_EMAIL]" in result["user"]["contact"]
        assert "[REDACTED_PHONE]" in result["user"]["phone"]

    def test_redact_list(self):
        data = ["user@test.com", "no-pii", "123-45-6789"]
        result = self.redactor.redact(data)
        assert "[REDACTED_EMAIL]" in result[0]
        assert result[1] == "no-pii"
        assert "[REDACTED_SSN]" in result[2]

    def test_redact_tuple_preserved(self):
        data = ("user@test.com", "safe")
        result = self.redactor.redact(data)
        assert isinstance(result, tuple)

    def test_redact_non_string(self):
        assert self.redactor.redact(42) == 42
        assert self.redactor.redact(None) is None
        assert self.redactor.redact(True) is True

    def test_extra_patterns(self):
        redactor = PIIRedactor(extra_patterns={"EMPLOYEE_ID": r"EMP-\d{6}"})
        text = "Employee EMP-123456 logged in"
        result = redactor.redact_text(text)
        assert "[REDACTED_EMPLOYEE_ID]" in result
        assert "EMP-123456" not in result

    def test_multiple_pii_in_one_string(self):
        text = "Email: a@b.com, Phone: 555-111-2222, SSN: 123-45-6789"
        result = self.redactor.redact_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_SSN]" in result
