from __future__ import annotations

import functools

from agent_chaos.chaos import (
    context_mutate,
    llm_rate_limit,
    llm_slow_ttft,
    llm_timeout,
    tool_error,
    tool_mutate,
)
from agent_chaos.scenario import (
    CompletesWithin,
    Scenario,
)

from agent import run_agent, run_agent_streaming

from .mutators import inject_conflicting_data, inject_distractor_message

multi_chaos_scenarios = [
    # Rate limit + slow TTFT
    Scenario(
        name="multi-ratelimit-and-slow-ttft",
        description="Tests agent resilience to multiple LLM failures: rate limits combined with slow time-to-first-token",
        agent=functools.partial(
            run_agent_streaming,
            query="What's the weather in Tokyo and Sydney?",
        ),
        chaos=[
            llm_rate_limit().on_call(3),
            llm_slow_ttft(delay=1.0),
        ],
        assertions=[
            CompletesWithin(120.0),
        ],
        meta={"kind": "multi", "chaos_types": ["rate_limit", "slow_ttft"]},
    ),
    # Tool error + LLM timeout on retry
    Scenario(
        name="multi-tool-error-then-timeout",
        description="Tests agent recovery from cascading failures: tool error followed by LLM timeout on retry",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Berlin?",
        ),
        chaos=[
            tool_error("Temporary error").for_tool("get_weather").on_call(1),
            llm_timeout().on_call(2),
        ],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "multi", "chaos_types": ["tool_error", "timeout"]},
    ),
    # Everything at once (chaos storm)
    Scenario(
        name="multi-chaos-storm",
        description="Tests agent resilience under extreme conditions: slow streaming, corrupted tool data, and context manipulation simultaneously",
        agent=functools.partial(
            run_agent,
            query="What's the weather everywhere?",
        ),
        chaos=[
            llm_slow_ttft(delay=0.5),
            tool_mutate(inject_conflicting_data),
            context_mutate(inject_distractor_message).on_call(2),
        ],
        assertions=[
            CompletesWithin(120.0),
        ],
        meta={
            "kind": "multi",
            "chaos_types": ["slow_ttft", "tool_mutate", "context_mutate"],
        },
    ),
]
