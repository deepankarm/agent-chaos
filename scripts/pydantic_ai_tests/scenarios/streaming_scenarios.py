from __future__ import annotations

import functools

from agent_chaos.chaos import (
    llm_slow_chunks,
    llm_slow_ttft,
    llm_stream_cut,
)
from agent_chaos.scenario import (
    CompletesWithin,
    ExpectError,
    MaxFailedCalls,
    Scenario,
)

from agent import run_agent_streaming

streaming_scenarios = [
    # Stream cut after 5 chunks
    Scenario(
        name="stream-cut-early",
        description="Tests agent recovery when streaming response is cut early (after 5 chunks)",
        agent=functools.partial(
            run_agent_streaming,
            query="What's the weather in Tokyo? Give me a detailed answer.",
        ),
        chaos=[llm_stream_cut(after_chunks=5)],
        assertions=[
            CompletesWithin(60.0),
            ExpectError(r"Connection|stream|terminated"),
        ],
        meta={"kind": "stream", "chaos_type": "cut", "trigger": "after_chunks(5)"},
    ),
    # Stream cut after 20 chunks (later in response)
    Scenario(
        name="stream-cut-late",
        description="Tests agent handling when streaming response is cut later in the stream (after 20 chunks)",
        agent=functools.partial(
            run_agent_streaming,
            query="What's the weather in all major cities?",
        ),
        chaos=[llm_stream_cut(after_chunks=20)],
        assertions=[
            CompletesWithin(60.0),
        ],
        meta={"kind": "stream", "chaos_type": "cut", "trigger": "after_chunks(20)"},
    ),
    # Slow TTFT (2 second delay)
    Scenario(
        name="stream-slow-ttft",
        description="Tests agent tolerance to slow time-to-first-token (2 second delay)",
        agent=functools.partial(
            run_agent_streaming,
            query="What's the weather in Sydney?",
        ),
        chaos=[llm_slow_ttft(delay=2.0)],
        assertions=[
            CompletesWithin(60.0),
            MaxFailedCalls(0),
        ],
        meta={"kind": "stream", "chaos_type": "slow_ttft", "delay": "2.0s"},
    ),
    # Slow chunks (0.5s between each)
    Scenario(
        name="stream-slow-chunks",
        description="Tests agent handling of slow streaming chunks (0.5s delay between chunks)",
        agent=functools.partial(
            run_agent_streaming,
            query="What's the weather in London?",
        ),
        chaos=[llm_slow_chunks(delay=0.5)],
        assertions=[
            CompletesWithin(120.0),
            MaxFailedCalls(0),
        ],
        meta={"kind": "stream", "chaos_type": "slow_chunks", "delay": "0.5s"},
    ),
]
