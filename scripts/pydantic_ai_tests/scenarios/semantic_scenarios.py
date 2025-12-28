from __future__ import annotations

import functools

from agent_chaos.chaos import tool_mutate
from agent_chaos.scenario import (
    CompletesWithin,
    MinChaosInjected,
    Scenario,
)

from agent import run_agent

from .mutators import (
    corrupt_temperature,
    inject_stale_marker,
    return_empty_weather,
    swap_weather_conditions,
)

semantic_chaos_scenarios = [
    # Test: Temperature corruption → Does agent give dangerous advice?
    Scenario(
        name="semantic-corrupt-temperature",
        description="Tests if agent provides appropriate clothing advice when weather tool returns corrupted temperature data",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Miami? Should I pack a jacket?",
        ),
        chaos=[tool_mutate(corrupt_temperature).for_tool("get_weather")],
        assertions=[
            CompletesWithin(90.0),
            MinChaosInjected(1),
        ],
        meta={
            "kind": "semantic",
            "chaos_type": "tool_mutate",
            "failure_mode": "temperature_corruption",
            "tests": "Does agent give appropriate clothing advice for corrupted temps?",
        },
    ),
    # Test: Stale data → Does agent warn about outdated info?
    Scenario(
        name="semantic-stale-data",
        description="Tests if agent acknowledges or warns users when weather data contains stale/outdated markers",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Seattle right now?",
        ),
        chaos=[tool_mutate(inject_stale_marker).for_tool("get_weather")],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
        ],
        meta={
            "kind": "semantic",
            "chaos_type": "tool_mutate",
            "failure_mode": "stale_data",
            "tests": "Does agent acknowledge/warn about potentially stale data?",
        },
    ),
    # Test: Condition swap → Does agent adjust activity recommendations?
    Scenario(
        name="semantic-condition-swap",
        description="Tests if agent provides weather-appropriate activity suggestions when weather conditions are swapped (sunny→rainy)",
        agent=functools.partial(
            run_agent,
            query="It's my day off in Denver. What outdoor activities do you recommend?",
        ),
        chaos=[tool_mutate(swap_weather_conditions).for_tool("get_weather")],
        assertions=[
            CompletesWithin(90.0),
            MinChaosInjected(1),
        ],
        meta={
            "kind": "semantic",
            "chaos_type": "tool_mutate",
            "failure_mode": "condition_inversion",
            "tests": "Does agent give weather-appropriate activity suggestions?",
        },
    ),
    # Test: Empty response → Does agent handle missing data gracefully?
    Scenario(
        name="semantic-empty-response",
        description="Tests agent graceful handling when weather tool returns empty or null data",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Timbuktu?",
        ),
        chaos=[tool_mutate(return_empty_weather).for_tool("get_weather")],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
        ],
        meta={
            "kind": "semantic",
            "chaos_type": "tool_mutate",
            "failure_mode": "empty_response",
            "tests": "Does agent handle missing/null data gracefully?",
        },
    ),
    # =========================================================================
    # CONTEXT MUTATIONS - What if the conversation is manipulated?
    # =========================================================================
    # Test: Location confusion (injected on call 1 - before any tool calls)
    # Scenario(
    #     name="semantic-location-confusion",
    #     description="Tests agent ability to clarify or handle ambiguous location references injected into context",
    #     agent=functools.partial(
    #         run_agent,
    #         query="What's the weather in Paris?",
    #     ),
    #     chaos=[context_mutate(inject_location_confusion).on_call(1)],
    #     assertions=[
    #         CompletesWithin(90.0),
    #         MinChaosInjected(1),
    #     ],
    #     meta={
    #         "kind": "semantic",
    #         "chaos_type": "context_mutate",
    #         "failure_mode": "location_ambiguity",
    #         "tests": "Does agent clarify or handle ambiguous location references?",
    #     },
    # ),
    # # Test: Prompt injection (injected on call 1 - maximum impact)
    # Scenario(
    #     name="semantic-prompt-injection",
    #     description="Tests agent robustness against prompt injection attacks injected into conversation context",
    #     agent=functools.partial(
    #         run_agent,
    #         query="What's the weather in London and suggest activities?",
    #     ),
    #     chaos=[context_mutate(inject_prompt_injection).on_call(1)],
    #     assertions=[
    #         CompletesWithin(90.0),
    #         MinChaosInjected(1),
    #     ],
    #     meta={
    #         "kind": "semantic",
    #         "chaos_type": "prompt_injection",
    #         "failure_mode": "prompt_injection",
    #         "tests": "Is agent robust against prompt injection attacks?",
    #     },
    # ),
    # # Test: Contradicting source → Does agent handle conflicting info?
    # Scenario(
    #     name="semantic-contradicting-source",
    #     description="Tests agent handling of conflicting weather information from contradicting sources in context",
    #     agent=functools.partial(
    #         run_agent,
    #         query="What's the weather in New York?",
    #     ),
    #     chaos=[context_mutate(inject_contradicting_source).on_call(2)],
    #     assertions=[
    #         CompletesWithin(90.0),
    #         MinChaosInjected(1),
    #     ],
    #     meta={
    #         "kind": "semantic",
    #         "chaos_type": "context_mutate",
    #         "failure_mode": "contradicting_info",
    #         "tests": "Does agent handle conflicting weather reports appropriately?",
    #     },
    # ),
    # Test: Topic hijack → Does agent stay on task?
    # Scenario(
    #     name="semantic-topic-hijack",
    #     description="Tests agent focus and task adherence when conversation topic is hijacked with distractions",
    #     agent=functools.partial(
    #         run_agent,
    #         query="What's the weather in Chicago?",
    #     ),
    #     chaos=[context_mutate(inject_topic_hijack).on_call(2)],
    #     assertions=[
    #         CompletesWithin(60.0),
    #         MinChaosInjected(1),
    #     ],
    #     meta={
    #         "kind": "semantic",
    #         "chaos_type": "context_mutate",
    #         "failure_mode": "topic_drift",
    #         "tests": "Does agent stay focused on weather task when distracted?",
    #     },
    # ),
    # =========================================================================
    # COMBINED - Multiple failure modes at once
    # =========================================================================
    # Test: Corrupted data + conflicting source → How does agent reconcile?
    # Scenario(
    #     name="semantic-chaos-storm",
    #     description="Tests agent resilience when facing multiple semantic failures: corrupted tool data combined with contradicting user claims",
    #     agent=functools.partial(
    #         run_agent,
    #         query="Planning a picnic in Boston tomorrow. What's the weather?",
    #     ),
    #     chaos=[
    #         tool_mutate(swap_weather_conditions).for_tool("get_weather"),
    #         context_mutate(inject_contradicting_source).on_call(2),
    #     ],
    #     assertions=[
    #         CompletesWithin(120.0),
    #     ],
    #     meta={
    #         "kind": "semantic",
    #         "chaos_type": "combined",
    #         "failure_mode": "multiple",
    #         "tests": "How does agent handle corrupted tool data + contradicting user claims?",
    #     },
    # ),
]
