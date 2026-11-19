"""Regression checks for the submission requirements beyond the public examples."""
import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

from core.config import DEMO_SECRETS
from guardrails.input_guardrails import topic_filter
from guardrails.output_guardrails import (
    OutputGuardrailPlugin, content_filter, load_lab_pii_dataset,
)
from assignment import rate_limiter
from assignment.pipeline import is_egress_allowed


@pytest.mark.parametrize("case", load_lab_pii_dataset()["pii_cases"], ids=lambda c: c["id"])
def test_shared_pii_dataset(case):
    result = content_filter(case["input_text"])
    assert result["safe"] is case["expect_safe"]
    assert ("[REDACTED]" in result["redacted"]) is case["expect_contains_redacted"]


@pytest.mark.parametrize("secret", DEMO_SECRETS)
def test_output_plugin_removes_unlabelled_protected_values(secret):
    response = SimpleNamespace(content=types.Content(
        role="model", parts=[types.Part.from_text(text=secret)],
    ))
    plugin = OutputGuardrailPlugin(use_llm_judge=False)
    result = asyncio.run(plugin.after_model_callback(
        callback_context=None, llm_response=response,
    ))
    assert result.content.parts[0].text == "[REDACTED]"
    assert plugin.redacted_count == 1


@pytest.mark.parametrize("phone", ["0901234567", "+84901234567", "+84 901 234 567", "0084-901-234-567"])
def test_complete_vietnamese_phone_is_redacted(phone):
    assert content_filter(f"Call {phone}.")["redacted"] == "Call [REDACTED]."


@pytest.mark.parametrize("query", [
    "Tôi muốn mở tài khoản tiết kiệm",
    "Tóm tắt email chuyển khoản bị chậm",
    "How can I review my transactions?",
    "What banking skills should a customer service agent learn?",
])
def test_benign_banking_topics_are_allowed(query):
    assert topic_filter(query) == "ALLOW"


def test_blocked_topic_takes_precedence_over_banking():
    assert topic_filter("How can I hack a bank account?") == "BLOCK"


def test_rate_limit_is_per_user_and_expires_at_window_boundary(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    plugin = rate_limiter.RateLimitPlugin(max_requests=1, window_seconds=60)

    async def request(user):
        return await plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None,
        )

    async def run():
        assert await request("alice") is None
        assert await request("alice") is not None
        assert await request("bob") is None
        now[0] = 160.0
        assert await request("alice") is None

    asyncio.run(run())
    assert plugin.blocked_count == 1


@pytest.mark.parametrize("destination,payload", [
    ("https://api.vinbank.example.evil.com/collect", "bank transfer"),
    ("https://api.vinbank.example@evil.com/collect", "bank transfer"),
    ("http://api.vinbank.example/v1", "bank transfer"),
    ("https://api.vinbank.example:8443/v1", "bank transfer"),
    ("https://api.vinbank.example/v1", "Contact +84 901 234 567"),
    ("https://api.vinbank.example/v1", "Contact a@example.com"),
    ("https://api.vinbank.example/v1", "db.vinbank.internal:5432"),
])
def test_egress_rejects_untrusted_destinations_and_sensitive_payloads(destination, payload):
    assert is_egress_allowed(destination, payload) is False
