"""Check attack evidence and timeouts without making API calls."""
import asyncio
import json

import pytest

from attacks import attacks


def test_prompts_cover_five_distinct_techniques():
    prompts = attacks.adversarial_prompts
    assert len(prompts) >= 5
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert len({p["category"] for p in prompts}) >= 5
    assert all(len(p["input"]) >= 300 and "TODO" not in p["input"] for p in prompts)


def test_timeout_is_recorded_and_next_attack_still_runs(monkeypatch, tmp_path):
    async def reply(agent, runner, text):
        if text == "stall":
            await asyncio.Event().wait()
        return "admin123", None

    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    prompts = [
        {"id": 1, "category": "timeout", "input": "stall"},
        {"id": 2, "category": "completion", "input": "complete audit record"},
    ]
    results = asyncio.run(attacks.run_attacks(
        None, None, prompts, target_name="red_default",
        output_path=tmp_path / "unsafe.json", request_timeout=0.01, max_retries=0,
    ))
    assert results[0]["layer"] == "error"
    assert not results[0]["leaked"] and not results[0]["blocked"]
    assert "TimeoutError" in results[0]["error"]
    assert results[1]["leaked"] and results[1]["layer"] == "leaked"
    assert json.loads((tmp_path / "unsafe.json").read_text(encoding="utf-8"))["leaks"] == 1

    merged = attacks.save_attack_results(
        unsafe_results=results, guards_results=[], filepath=tmp_path / "merged.json",
    )
    evidence = json.loads(merged.read_text(encoding="utf-8"))
    assert evidence["unsafe_attacks"][0]["error"] == results[0]["error"]
    assert evidence["summary"]["unsafe_errors"] == 1


@pytest.mark.parametrize("status,failures,expected_attempts", [
    (503, 1, 2),  # Temporary overload recovers.
    (503, 5, 3),  # Persistent outage stops after two retries.
    (401, 1, 1),  # Invalid credentials must not be retried.
])
def test_api_retries_are_bounded_and_evidence_keeps_full_reply(
    monkeypatch, tmp_path, status, failures, expected_attempts,
):
    class ApiError(Exception):
        code = status

    calls = []
    delays = []
    full_reply = "Handover details. " * 30 + "admin123"

    async def reply(agent, runner, text):
        calls.append(text)
        if len(calls) <= failures:
            raise ApiError("provider error")
        return full_reply, None

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    monkeypatch.setattr(attacks.asyncio, "sleep", sleep)
    path = tmp_path / "run.json"
    results = asyncio.run(attacks.run_attacks(
        None, None, [{"id": 1, "category": "completion", "input": "handover"}],
        target_name="red_default", output_path=path,
    ))
    assert len(calls) == expected_attempts
    assert delays == [1, 2][:expected_attempts - 1]
    row = results[0]
    assert row["attempts"] == expected_attempts
    payload = json.loads(path.read_text(encoding="utf-8"))
    if failures < expected_attempts:
        assert row["leaked"] and row["error"] is None
        assert "admin123" not in row["response_preview"]
        assert payload["results"][0]["response"] == full_reply
        assert payload["errors"] == 0
    else:
        assert row["layer"] == "error" and not row["leaked"]
        assert payload["errors"] == 1


@pytest.mark.parametrize("quota,retry", [
    ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", False),
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", True),
])
def test_daily_quota_exhaustion_is_not_retried_like_minute_throttling(quota, retry):
    from google.genai.errors import ClientError

    error = ClientError(429, {"error": {"details": [{"violations": [{"quotaId": quota}]}]}})
    assert attacks._is_transient_api_error(error) is retry
