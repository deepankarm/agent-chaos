"""Chaos types and factories.

All chaos goes in one list. System routes automatically.

Usage:
    from agent_chaos import (
        chaos_context,
        user_input_mutate,
        llm_rate_limit, llm_timeout, llm_stream_cut,
        tool_error, tool_mutate,
        context_mutate,
        history_mutate, history_truncate, history_inject,
    )

    with chaos_context(
        name="test",
        chaos=[
            user_input_mutate(inject_typos),
            llm_rate_limit().after_calls(2),
            llm_stream_cut(after_chunks=10),
            tool_error("down").for_tool("weather"),
        ],
    ) as ctx:
        ...

Multi-turn support:
    # Turn-based triggers
    llm_rate_limit().on_turn(2)          # Fire on turn 2
    llm_rate_limit().after_turns(1)      # Fire after turn 1
    tool_error("down").on_turn(2).after_calls(1)  # Combined

    # Between-turn history chaos
    history_mutate(fn).between_turns(1, 2)
    history_truncate(keep_last=3).between_turns(2, 3)
"""

from agent_chaos.chaos.base import Chaos, ChaosPoint, ChaosResult
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.chaos.context import ContextChaos, context_mutate
from agent_chaos.chaos.history import (
    HistoryInjectChaos,
    HistoryMutateChaos,
    HistoryTruncateChaos,
    history_inject,
    history_mutate,
    history_truncate,
)
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
from agent_chaos.chaos.user import UserInputChaos, user_input_mutate

__all__ = [
    # Base
    "Chaos",
    "ChaosPoint",
    "ChaosResult",
    "ChaosBuilder",
    # User input chaos
    "UserInputChaos",
    "user_input_mutate",
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
    # History chaos (between-turn)
    "HistoryMutateChaos",
    "HistoryTruncateChaos",
    "HistoryInjectChaos",
    "history_mutate",
    "history_truncate",
    "history_inject",
]
