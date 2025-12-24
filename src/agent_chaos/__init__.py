"""agent-chaos — Chaos Engineering for AI Agents

Introduce a little chaos at every boundary of your agent.
"""

from agent_chaos.core.context import ChaosContext, chaos_context
from agent_chaos.chaos import (
    # Base
    Chaos,
    ChaosPoint,
    ChaosResult,
    ChaosBuilder,
    # LLM chaos
    llm_rate_limit,
    llm_timeout,
    llm_server_error,
    llm_auth_error,
    llm_context_length,
    # Stream chaos
    llm_stream_cut,
    llm_stream_hang,
    llm_slow_ttft,
    llm_slow_chunks,
    # Tool chaos
    tool_error,
    tool_empty,
    tool_timeout,
    tool_mutate,
    # Context chaos
    context_mutate,
)

__all__ = [
    # Context manager
    "chaos_context",
    "ChaosContext",
    # Base
    "Chaos",
    "ChaosPoint",
    "ChaosResult",
    "ChaosBuilder",
    # LLM chaos
    "llm_rate_limit",
    "llm_timeout",
    "llm_server_error",
    "llm_auth_error",
    "llm_context_length",
    # Stream chaos
    "llm_stream_cut",
    "llm_stream_hang",
    "llm_slow_ttft",
    "llm_slow_chunks",
    # Tool chaos
    "tool_error",
    "tool_empty",
    "tool_timeout",
    "tool_mutate",
    # Context chaos
    "context_mutate",
]
