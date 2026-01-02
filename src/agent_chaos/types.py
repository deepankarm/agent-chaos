"""Core type definitions for agent-chaos.

This module contains enums, type aliases, and base types used across the codebase.
Provider-specific types (Anthropic, OpenAI, Gemini) are kept in patch/providers/
to avoid leaking SDK dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent_chaos.scenario.model import TurnResult


class ChaosPoint(str, Enum):
    """Injection points for chaos."""

    USER_INPUT = "user_input"  # Before agent processes -> mutate user query
    LLM_CALL = "llm_call"  # Before LLM call -> raise exception
    STREAM = "stream"  # During streaming -> hang/cut/slow
    TOOL_RESULT = "tool_result"  # After tool returns -> mutate result
    MESSAGES = "messages"  # Before LLM call -> mutate messages array (RAG/memory)


class ChaosAction(str, Enum):
    """Actions that chaos can take."""

    PROCEED = "proceed"  # Continue without modification
    RAISE = "raise"  # Raise an exception
    MUTATE = "mutate"  # Return a mutated value
    HANG = "hang"  # Hang/block (for stream chaos)
    DELAY = "delay"  # Add delay (for stream chaos)


# Type alias for dynamic turn input generators
TurnInputGenerator = Callable[["list[TurnResult]"], str]

# Type alias for turn input (static string or dynamic generator)
TurnInput = str | TurnInputGenerator
