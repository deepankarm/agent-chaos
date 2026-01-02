"""Pydantic event models for agent-chaos.

All events emitted by the system are defined here with a consistent schema.
This enables interoperability with other tools and languages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    """Base class for all events."""

    timestamp: datetime = Field(default_factory=_utc_now)
    trace_id: str = ""
    trace_name: str = ""
    span_id: str = ""
    provider: str = ""

    model_config = {"extra": "allow"}


class TraceStartEvent(BaseEvent):
    """Emitted when a chaos session (trace) starts."""

    type: Literal["trace_start"] = "trace_start"


class TraceEndEvent(BaseEvent):
    """Emitted when a chaos session (trace) ends."""

    type: Literal["trace_end"] = "trace_end"
    total_calls: int = 0
    failed_calls: int = 0
    fault_count: int = 0
    success: bool = True
    error: str | None = None
    duration_s: float | None = None


class SpanStartEvent(BaseEvent):
    """Emitted when an LLM call (span) starts."""

    type: Literal["span_start"] = "span_start"


class SpanEndEvent(BaseEvent):
    """Emitted when an LLM call (span) ends."""

    type: Literal["span_end"] = "span_end"
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class FaultInjectedEvent(BaseEvent):
    """Emitted when chaos is injected."""

    type: Literal["fault_injected"] = "fault_injected"
    fault_type: str = ""
    chaos_point: str = ""  # LLM, STREAM, TOOL, CONTEXT, USER_INPUT
    chaos_fn_name: str | None = None
    chaos_fn_doc: str | None = None
    target_tool: str | None = None
    original: str | None = None
    mutated: str | None = None
    added_messages: list[dict[str, Any]] | None = None
    removed_messages: list[dict[str, Any]] | None = None
    added_count: int | None = None
    removed_count: int | None = None


class TTFTEvent(BaseEvent):
    """Emitted when time-to-first-token is recorded."""

    type: Literal["ttft"] = "ttft"
    ttft_ms: float = 0.0
    is_delayed: bool = False


class StreamCutEvent(BaseEvent):
    """Emitted when a stream is cut."""

    type: Literal["stream_cut"] = "stream_cut"
    chunk_count: int = 0


class StreamStatsEvent(BaseEvent):
    """Emitted with final stream statistics."""

    type: Literal["stream_stats"] = "stream_stats"
    chunk_count: int = 0


class TokenUsageEvent(BaseEvent):
    """Emitted when token usage is recorded."""

    type: Literal["token_usage"] = "token_usage"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0


class ToolUseEvent(BaseEvent):
    """Emitted when the LLM requests a tool use."""

    type: Literal["tool_use"] = "tool_use"
    tool_name: str = ""
    tool_use_id: str | None = None
    input_bytes: int | None = None
    args: dict[str, Any] | None = None


class ToolStartEvent(BaseEvent):
    """Emitted when tool execution starts."""

    type: Literal["tool_start"] = "tool_start"
    tool_name: str = ""
    tool_use_id: str | None = None
    input_bytes: int | None = None
    llm_args_ms: float | None = None


class ToolEndEvent(BaseEvent):
    """Emitted when tool execution ends."""

    type: Literal["tool_end"] = "tool_end"
    tool_name: str = ""
    tool_use_id: str | None = None
    success: bool = True
    duration_ms: float | None = None
    output_bytes: int | None = None
    result: str | None = None
    error: str | None = None
    resolved_in_call_id: str | None = None


# Union of all event types for type checking
Event = Annotated[
    Union[
        TraceStartEvent,
        TraceEndEvent,
        SpanStartEvent,
        SpanEndEvent,
        FaultInjectedEvent,
        TTFTEvent,
        StreamCutEvent,
        StreamStatsEvent,
        TokenUsageEvent,
        ToolUseEvent,
        ToolStartEvent,
        ToolEndEvent,
    ],
    Field(discriminator="type"),
]
