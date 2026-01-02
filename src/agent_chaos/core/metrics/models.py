"""Sub-models for MetricsStore.

These models organize metrics data into logical groups:
- CallStats: Call counting and latency tracking
- TokenStats: Token accumulation
- StreamStats: Stream event tracking (ttft, hangs, cuts)
- ToolTracking: Tool execution state
- ConversationState: Conversation timeline and turn state
- ActiveCallInfo: Per-call tracking during execution
- CallRecord: Completed call data
- FaultRecord: Injected fault data
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CallStats(BaseModel):
    """Call counting and latency statistics."""

    count: int = 0
    retries: int = 0
    by_provider: dict[str, int] = Field(default_factory=dict)
    latencies: list[float] = Field(default_factory=list)


class TokenStats(BaseModel):
    """Cumulative token tracking."""

    input: int = 0
    output: int = 0


class StreamStats(BaseModel):
    """Stream event tracking."""

    ttft_times: list[float] = Field(default_factory=list)
    hang_events: list[int] = Field(default_factory=list)
    stream_cuts: list[int] = Field(default_factory=list)
    corruption_events: list[int] = Field(default_factory=list)
    chunk_counts: list[int] = Field(default_factory=list)


class ToolTracking(BaseModel):
    """Tool execution state tracking."""

    use_to_call_id: dict[str, str] = Field(default_factory=dict)
    use_to_name: dict[str, str] = Field(default_factory=dict)
    started_at: dict[str, float] = Field(default_factory=dict)
    ended: set[str] = Field(default_factory=set)
    in_conversation: set[str] = Field(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConversationState(BaseModel):
    """Conversation timeline and turn state."""

    entries: list[dict[str, Any]] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.monotonic)
    user_message_recorded: bool = False
    current_turn: int = 0
    system_prompt: str | None = None
    system_prompt_recorded: bool = False


class ActiveCallInfo(BaseModel):
    """Per-call tracking during execution."""

    provider: str
    start_time: float
    call_id: str
    usage: dict[str, Any] = Field(default_factory=dict)
    tool_uses: list[dict[str, Any]] = Field(default_factory=list)
    stream_chunks: int = 0


class CallRecord(BaseModel):
    """Completed call record."""

    call_id: str
    provider: str
    success: bool
    latency: float
    error: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    tool_uses: list[dict[str, Any]] = Field(default_factory=list)
    stream_chunks: int = 0


class FaultRecord(BaseModel):
    """Injected fault record."""

    call_id: str
    fault_type: str
