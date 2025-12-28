from __future__ import annotations

import functools

from agent_chaos.chaos import (
    llm_auth_error,
    llm_context_length,
    llm_rate_limit,
    llm_server_error,
    llm_timeout,
)
from agent_chaos.scenario import (
    CompletesWithin,
    ExpectError,
    MinChaosInjected,
    MinLLMCalls,
    Scenario,
)

from agent import run_agent

llm_scenarios = [
    # Rate limit on call 2
    Scenario(
        name="llm-rate-limit-call-2",
        description="Tests agent resilience to rate limit errors on the second LLM call",
        agent=functools.partial(
            run_agent,
            query="What's the weather in London and Tokyo?",
        ),
        chaos=[llm_rate_limit(retry_after=1.0).on_call(2)],
        assertions=[
            CompletesWithin(60.0),
            MinLLMCalls(1),
            MinChaosInjected(1),
            ExpectError(r"RateLimit|429|rate limit"),
        ],
        meta={"kind": "llm", "chaos_type": "rate_limit", "trigger": "on_call(2)"},
    ),
    # Rate limit after 2 calls
    Scenario(
        name="llm-rate-limit-after-2",
        description="Tests agent handling of rate limits after completing 2 successful LLM calls",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Paris, Berlin, and Sydney?",
        ),
        chaos=[llm_rate_limit().after_calls(2)],
        assertions=[
            CompletesWithin(60.0),
            MinLLMCalls(2),
            MinChaosInjected(1),
            ExpectError(r"RateLimit|429|rate limit"),
        ],
        meta={
            "kind": "llm",
            "chaos_type": "rate_limit",
            "trigger": "after_calls(2)",
        },
    ),
    # Timeout on first call
    Scenario(
        name="llm-timeout-first-call",
        description="Tests agent recovery from timeout errors on the initial LLM call",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Sydney?",
        ),
        chaos=[llm_timeout(5.0).on_call(1)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"Timeout|APITimeout|timed out"),
        ],
        meta={"kind": "llm", "chaos_type": "timeout", "trigger": "on_call(1)"},
    ),
    # Timeout with probability
    Scenario(
        name="llm-timeout-probabilistic",
        description="Tests agent behavior with probabilistic timeout failures (80% chance)",
        agent=functools.partial(
            run_agent,
            query="What's the weather in New York?",
        ),
        chaos=[llm_timeout().with_probability(0.8)],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={
            "kind": "llm",
            "chaos_type": "timeout",
            "trigger": "probability(0.8)",
        },
    ),
    # Server error (500)
    Scenario(
        name="llm-server-error",
        description="Tests agent resilience to LLM provider server errors (500) on the first call",
        agent=functools.partial(
            run_agent,
            query="What's the weather in London?",
        ),
        chaos=[llm_server_error("Temporary outage").on_call(1)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"InternalServer|500|server error"),
        ],
        meta={"kind": "llm", "chaos_type": "server_error", "trigger": "on_call(1)"},
    ),
    # Auth error (401)
    Scenario(
        name="llm-auth-error",
        description="Tests agent handling of authentication errors (401) from LLM provider",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Paris?",
        ),
        chaos=[llm_auth_error("Invalid API key").on_call(1)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"Authentication|401|auth|invalid"),
        ],
        meta={"kind": "llm", "chaos_type": "auth_error", "trigger": "on_call(1)"},
    ),
    # Context length exceeded
    Scenario(
        name="llm-context-length",
        description="Tests agent behavior when LLM context length limit is exceeded",
        agent=functools.partial(
            run_agent,
            query="What's the weather in Berlin?",
        ),
        chaos=[llm_context_length("Context too long").on_call(1)],
        assertions=[
            CompletesWithin(60.0),
            MinChaosInjected(1),
            ExpectError(r"context|length|too long|invalid"),
        ],
        meta={
            "kind": "llm",
            "chaos_type": "context_length",
            "trigger": "on_call(1)",
        },
    ),
]
