"""An API outage must not be reported as a completed checkpoint."""
import asyncio

import pytest

import main
from agents import agent, guards_agent
from attacks import attacks


@pytest.mark.parametrize("case", ["success", "api_error", "no_leak"])
def test_checkpoint4_saves_evidence_before_reporting_status(monkeypatch, case):
    monkeypatch.setattr(agent, "create_red_agent_default", lambda: (None, None))
    monkeypatch.setattr(guards_agent, "create_red_agent_advance", lambda: (None, None))
    saved = []

    async def run(*args, target_name):
        red = target_name == "red_default"
        return [dict(
            leaked=red and case != "no_leak",
            layer="error" if case == "api_error" and red else "input_injection",
            error="503" if case == "api_error" and red else None,
        ) for _ in range(5)]

    monkeypatch.setattr(attacks, "run_attacks", run)
    monkeypatch.setattr(attacks, "save_attack_results", lambda **kwargs: saved.append(kwargs))
    if case == "success":
        assert len(asyncio.run(main.part4_attacks())["red_default"]) == 5
    else:
        with pytest.raises(RuntimeError, match="Checkpoint 4"):
            asyncio.run(main.part4_attacks())
    assert len(saved) == 1
    assert len(saved[0]["unsafe_results"]) == len(saved[0]["guards_results"]) == 5
