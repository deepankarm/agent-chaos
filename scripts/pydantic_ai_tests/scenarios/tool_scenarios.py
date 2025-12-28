from __future__ import annotations

import functools

from agent_chaos.chaos import tool_empty, tool_error, tool_mutate, tool_timeout
from agent_chaos.scenario import (
    CompletesWithin,
    MaxFailedCalls,
    MinLLMCalls,
    Scenario,
)

from agent import run_agent

from .mutators import (
    add_noise_to_activity,
    corrupt_weather_data,
    inject_conflicting_data,
)

tool_scenarios = [
    # Tool returns error
    Scenario(
        name="tool-error-weather",
        description="Tests agent handling when the weather tool returns an error",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Tokyo?",
        ),
        chaos=[tool_error("Weather service unavailable").for_tool("get_weather")],
        assertions=[
            CompletesWithin(60.0),
            MinLLMCalls(1),
        ],
        meta={"kind": "tool", "chaos_type": "error", "target": "get_weather"},
    ),
    # Tool returns empty
    Scenario(
        name="tool-empty-weather",
        description="Tests agent behavior when weather tool returns empty/null response",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Paris?",
        ),
        chaos=[tool_empty().for_tool("get_weather")],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "tool", "chaos_type": "empty", "target": "get_weather"},
    ),
    # Tool timeout message
    Scenario(
        name="tool-timeout-activity",
        description="Tests agent resilience when activity suggestion tool times out",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Berlin?",
        ),
        chaos=[tool_timeout(timeout_seconds=10.0).for_tool("suggest_activity")],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={
            "kind": "tool",
            "chaos_type": "timeout",
            "target": "suggest_activity",
        },
    ),
    # Custom mutation: corrupt weather data
    Scenario(
        name="tool-mutate-corrupt-weather",
        description="Tests agent handling of corrupted weather data from tool responses",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Tokyo?",
        ),
        chaos=[tool_mutate(corrupt_weather_data)],
        assertions=[
            CompletesWithin(60.0),
            MaxFailedCalls(0),
        ],
        meta={
            "kind": "tool",
            "chaos_type": "mutate",
            "mutation": "corrupt_weather",
        },
    ),
    # Custom mutation: inject conflicting data
    Scenario(
        name="tool-mutate-conflicting-data",
        description="Tests agent behavior when tool returns conflicting or contradictory data",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Sydney?",
        ),
        chaos=[tool_mutate(inject_conflicting_data)],
        assertions=[
            CompletesWithin(60.0),
            MaxFailedCalls(0),
        ],
        meta={
            "kind": "tool",
            "chaos_type": "mutate",
            "mutation": "conflicting_data",
        },
    ),
    # Custom mutation: add noise to activity suggestions
    Scenario(
        name="tool-mutate-noisy-activity",
        description="Tests agent handling of noisy or malformed activity suggestion data",
        agent=functools.partial(
            run_agent,
            query="What's the weather in London?",
        ),
        chaos=[tool_mutate(add_noise_to_activity)],
        assertions=[
            CompletesWithin(60.0),
            MaxFailedCalls(0),
        ],
        meta={"kind": "tool", "chaos_type": "mutate", "mutation": "noisy_activity"},
    ),
    # Error on ALL tools
    Scenario(
        name="tool-error-all-tools",
        description="Tests agent behavior when all tools fail simultaneously",
        agent=functools.partial(
            run_agent,
            query="What's the weather in New York?",
        ),
        chaos=[
            tool_error("Service down").for_tool("get_weather"),
            tool_error("Service down").for_tool("suggest_activity"),
        ],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "tool", "chaos_type": "error", "target": "all_tools"},
    ),
]
