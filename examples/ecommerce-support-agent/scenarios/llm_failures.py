"""LLM failure scenarios - testing agent resilience to LLM API issues.

These scenarios inject LLM-level chaos like rate limits, timeouts,
and stream interruptions to test recovery behavior.
All scenarios use the Turn-based architecture.
"""

from __future__ import annotations

from agent_chaos import Turn
from agent_chaos.chaos import llm_rate_limit, llm_server_error, llm_timeout
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    ExpectError,
    MaxTotalLLMCalls,
    MinChaosInjected,
    Scenario,
)

from agent import run_support_agent

llm_failure_scenarios = [
    Scenario(
        name="llm-rate-limit-early",
        description="Rate limit hits after first LLM call - tests retry behavior",
        agent=run_support_agent,
        turns=[
            Turn("What's the status of order ORD-67890 and when will it arrive?"),
        ],
        chaos=[llm_rate_limit().after_calls(1)],
        assertions=[MinChaosInjected(1), MaxTotalLLMCalls(10)],
        tags=["llm_failure", "rate_limit", "early"],
    ),
    Scenario(
        name="llm-rate-limit-during-tools",
        description="Rate limit during multi-tool operation - tests mid-operation recovery",
        agent=run_support_agent,
        turns=[
            Turn(
                "Check if I can get a refund for order ORD-67890 and process it if eligible"
            )
        ],
        chaos=[llm_rate_limit().after_calls(2)],
        assertions=[MinChaosInjected(1)],
        tags=["llm_failure", "rate_limit", "mid_operation"],
    ),
    Scenario(
        name="llm-timeout",
        description="LLM request times out - tests timeout handling",
        agent=run_support_agent,
        turns=[Turn("I need help with my order ORD-12345")],
        chaos=[llm_timeout(delay=5.0).on_call(1)],
        assertions=[MinChaosInjected(1)],
        tags=["llm_failure", "timeout"],
    ),
    Scenario(
        name="llm-server-error",
        description="LLM returns 500 error - tests error recovery",
        agent=run_support_agent,
        turns=[Turn("Where is my package with tracking 1Z999AA10123456784?")],
        chaos=[llm_server_error().on_call(1)],
        assertions=[MinChaosInjected(1)],
        tags=["llm_failure", "server_error"],
    ),
    Scenario(
        name="llm-rate-limit-persistent",
        description="Persistent rate limiting - tests if agent eventually fails gracefully",
        agent=run_support_agent,
        turns=[Turn("Help me track my order")],
        chaos=[llm_rate_limit().always()],
        assertions=[ExpectError("rate.*limit|429"), MinChaosInjected(1)],
        tags=["llm_failure", "persistent_rate_limit"],
    ),
    Scenario(
        name="llm-rate-limit-intermittent",
        description="Intermittent rate limiting (50% chance) - tests flaky API behavior",
        agent=run_support_agent,
        turns=[
            Turn("Can you check order ORD-67890 and tell me about the shipping status?")
        ],
        chaos=[llm_rate_limit().with_probability(0.5)],
        assertions=[CompletesWithin(120.0)],
        tags=["llm_failure", "intermittent"],
    ),
    Scenario(
        name="llm-rate-limit-turn-2",
        description="Rate limit on second turn - tests multi-turn rate limit recovery",
        agent=run_support_agent,
        turns=[
            Turn("What's the status of order ORD-67890?"),
            Turn("Can you also check the shipping?"),
        ],
        chaos=[llm_rate_limit().on_turn(2).after_calls(1)],
        assertions=[AllTurnsComplete(allow_failures=1), MaxTotalLLMCalls(15)],
        tags=["llm_failure", "rate_limit_turn_2"],
    ),
]
