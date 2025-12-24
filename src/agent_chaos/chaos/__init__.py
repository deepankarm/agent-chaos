"""Chaos types and factories.

All chaos goes in one list. System routes automatically.

Usage:
    from agent_chaos import (
        chaos_context,
        llm_rate_limit, llm_timeout, llm_stream_cut,
        tool_error, tool_mutate,
        context_mutate,
    )

    with chaos_context(
        name="test",
        chaos=[
            llm_rate_limit().after_calls(2),
            llm_stream_cut(after_chunks=10),
            tool_error("down").for_tool("weather"),
        ],
    ) as ctx:
        ...
"""

from agent_chaos.chaos.base import Chaos, ChaosPoint, ChaosResult
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.chaos.llm import (
    LLMChaos,
    llm_auth_error,
    llm_context_length,
    llm_rate_limit,
    llm_server_error,
    llm_timeout,
)
from agent_chaos.chaos.stream import (
    StreamChaos,
    llm_slow_chunks,
    llm_slow_ttft,
    llm_stream_cut,
    llm_stream_hang,
)
from agent_chaos.chaos.tool import (
    ToolChaos,
    tool_empty,
    tool_error,
    tool_mutate,
    tool_timeout,
)
from agent_chaos.chaos.context import (
    ContextChaos,
    context_mutate,
)

__all__ = [
    # Base
    "Chaos",
    "ChaosPoint",
    "ChaosResult",
    "ChaosBuilder",
    # LLM chaos
    "LLMChaos",
    "llm_rate_limit",
    "llm_timeout",
    "llm_server_error",
    "llm_auth_error",
    "llm_context_length",
    # Stream chaos
    "StreamChaos",
    "llm_stream_cut",
    "llm_stream_hang",
    "llm_slow_ttft",
    "llm_slow_chunks",
    # Tool chaos
    "ToolChaos",
    "tool_error",
    "tool_empty",
    "tool_timeout",
    "tool_mutate",
    # Context chaos
    "ContextChaos",
    "context_mutate",
]
