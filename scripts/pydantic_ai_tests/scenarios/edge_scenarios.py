from __future__ import annotations

import functools

from agent_chaos.chaos import llm_rate_limit, llm_server_error
from agent_chaos.scenario import (
    CompletesWithin,
    ExpectError,
    MinChaosInjected,
    Scenario,
)

from agent import run_agent

edge_case_scenarios = [
    # Very first call fails
    Scenario(
        name="edge-first-call-fails",
        description="Tests agent recovery when the very first LLM call fails with a server error",
        agent=functools.partial(
            run_agent,
            query="What's the weather?",
        ),
        chaos=[llm_server_error().on_call(1)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"server|500"),
        ],
        meta={"kind": "edge", "chaos_type": "first_call_fail"},
    ),
    # Rate limit with probability 1.0 (always)
    Scenario(
        name="edge-always-rate-limit",
        description="Tests agent behavior when rate limits occur on every LLM call (100% probability)",
        agent=functools.partial(
            run_agent,
            query="Weather?",
        ),
        chaos=[llm_rate_limit().with_probability(1.0)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"RateLimit"),
        ],
        meta={"kind": "edge", "chaos_type": "always_fail"},
    ),
    # Very low probability (should usually pass)
    Scenario(
        name="edge-low-probability",
        description="Tests agent behavior with low-probability failures (10% chance of rate limit)",
        agent=functools.partial(
            run_agent,
            query="What's the weather in London?",
        ),
        chaos=[llm_rate_limit().with_probability(0.1)],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "edge", "chaos_type": "low_probability"},
    ),
]
