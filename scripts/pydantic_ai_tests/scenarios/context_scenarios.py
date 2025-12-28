from __future__ import annotations

import functools

from agent_chaos.chaos import context_mutate
from agent_chaos.scenario import (
    CompletesWithin,
    Scenario,
)

from agent import run_agent

from .mutators import inject_distractor_message, truncate_history

context_scenarios = [
    # Inject distractor message
    Scenario(
        name="context-inject-distractor",
        description="Tests agent resilience when distractor messages are injected into conversation context",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Tokyo?",
        ),
        chaos=[context_mutate(inject_distractor_message).on_call(2)],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "context", "chaos_type": "mutate", "mutation": "distractor"},
    ),
    # Truncate conversation history
    Scenario(
        name="context-truncate-history",
        description="Tests agent behavior when conversation history is truncated mid-conversation",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Paris and London?",
        ),
        chaos=[context_mutate(truncate_history).after_calls(1)],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "context", "chaos_type": "mutate", "mutation": "truncate"},
    ),
]
