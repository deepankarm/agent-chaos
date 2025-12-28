"""Baseline scenario - no chaos, establishes normal behavior."""

from __future__ import annotations

import functools

from agent_chaos.scenario import (
    CompletesWithin,
    MaxFailedCalls,
    MaxLLMCalls,
    MinLLMCalls,
    Scenario,
)

from agent import run_agent

baseline_scenario = Scenario(
    name="baseline-no-chaos",
    description="Baseline scenario with no chaos injection to establish normal agent behavior and performance metrics",
    agent=functools.partial(
        run_agent,
        query="What's the weather in Tokyo and what should I do today?",
    ),
    chaos=[],
    assertions=[
        CompletesWithin(60.0),
        MaxFailedCalls(0),
        MinLLMCalls(2),
        MaxLLMCalls(4),
    ],
    meta={"kind": "baseline", "chaos_type": "none"},
)
