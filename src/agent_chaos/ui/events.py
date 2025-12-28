"""Event bus for real-time UI updates with trace/span model."""

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    # Trace-level events
    TRACE_START = "trace_start"
    TRACE_END = "trace_end"

    # Span-level events (LLM calls)
    SPAN_START = "span_start"
    SPAN_END = "span_end"

    # Events within spans
    FAULT_INJECTED = "fault_injected"
    TTFT = "ttft"
    STREAM_CHUNK = "stream_chunk"
    STREAM_CUT = "stream_cut"
    STREAM_STATS = "stream_stats"
    TOKEN_USAGE = "token_usage"
    TOOL_USE = "tool_use"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"


@dataclass
class Event:
    type: EventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trace_id: str = ""
    trace_name: str = ""
    span_id: str = ""
    provider: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d)


@dataclass
class Span:
    """Represents an LLM call span within a trace."""

    span_id: str
    trace_id: str
    provider: str
    start_time: float
    end_time: float | None = None
    success: bool | None = None
    latency_ms: float | None = None
    error: str = ""
    events: list[Event] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.end_time is None:
            return "running"
        return "success" if self.success else "error"


@dataclass
class Trace:
    """Represents a chaos context session."""

    trace_id: str
    name: str
    description: str = ""
    start_time: str = ""
    end_time: str | None = None
    spans: list[Span] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.end_time is None:
            return "running"
        failed = sum(1 for s in self.spans if s.success is False)
        return "error" if failed > 0 else "success"

    @property
    def total_calls(self) -> int:
        return len(self.spans)

    @property
    def failed_calls(self) -> int:
        return sum(1 for s in self.spans if s.success is False)

    @property
    def fault_count(self) -> int:
        count = 0
        for span in self.spans:
            count += sum(1 for e in span.events if e.type == EventType.FAULT_INJECTED)
        return count


class EventBus:
    """Pub/sub event bus with trace/span model."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._traces: dict[str, Trace] = {}
        self._active_trace_id: str = ""
        self._active_spans: dict[str, Span] = {}

    def start_session(self, name: str, description: str = "") -> str:
        """Start a new trace (chaos session)."""
        trace_id = str(uuid.uuid4())[:8]
        self._active_trace_id = trace_id

        trace = Trace(
            trace_id=trace_id,
            name=name,
            description=description,
            start_time=datetime.now().isoformat(),
        )
        self._traces[trace_id] = trace

        self._emit(
            Event(
                type=EventType.TRACE_START,
                trace_id=trace_id,
                trace_name=name,
            )
        )
        return trace_id

    def end_session(self):
        """End the current trace."""
        if self._active_trace_id and self._active_trace_id in self._traces:
            trace = self._traces[self._active_trace_id]
            trace.end_time = datetime.now().isoformat()

            self._emit(
                Event(
                    type=EventType.TRACE_END,
                    trace_id=self._active_trace_id,
                    trace_name=trace.name,
                    data={
                        "total_calls": trace.total_calls,
                        "failed_calls": trace.failed_calls,
                        "fault_count": trace.fault_count,
                    },
                )
            )
            self._active_trace_id = ""

    @property
    def trace_id(self) -> str:
        return self._active_trace_id

    @property
    def session_id(self) -> str:
        """Alias for trace_id for backwards compat."""
        return self._active_trace_id

    def _emit(self, event: Event):
        """Emit an event to all subscribers."""
        if not event.trace_id:
            event.trace_id = self._active_trace_id
        if not event.trace_name and event.trace_id in self._traces:
            event.trace_name = self._traces[event.trace_id].name

        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def emit_call_start(self, call_id: str, provider: str):
        """Start a new span (LLM call)."""
        span = Span(
            span_id=call_id,
            trace_id=self._active_trace_id,
            provider=provider,
            start_time=time.monotonic(),
        )
        self._active_spans[call_id] = span

        if self._active_trace_id in self._traces:
            self._traces[self._active_trace_id].spans.append(span)

        self._emit(
            Event(
                type=EventType.SPAN_START,
                span_id=call_id,
                provider=provider,
            )
        )

    def emit_call_end(
        self,
        call_id: str,
        provider: str,
        success: bool,
        latency: float,
        error: str = "",
    ):
        """End a span (LLM call)."""
        if call_id in self._active_spans:
            span = self._active_spans.pop(call_id)
            span.end_time = time.monotonic()
            span.success = success
            span.latency_ms = latency * 1000
            span.error = error

        self._emit(
            Event(
                type=EventType.SPAN_END,
                span_id=call_id,
                provider=provider,
                data={
                    "success": success,
                    "latency_ms": latency * 1000,
                    "error": error,
                },
            )
        )

    def emit_fault(self, call_id: str, fault_type: str, provider: str):
        """Emit fault injection event."""
        event = Event(
            type=EventType.FAULT_INJECTED,
            span_id=call_id,
            provider=provider,
            data={"fault_type": fault_type},
        )

        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)

        self._emit(event)

    def emit_ttft(self, call_id: str, ttft: float):
        """Emit time-to-first-token event."""
        event = Event(
            type=EventType.TTFT,
            span_id=call_id,
            data={"ttft_ms": ttft * 1000},
        )

        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)

        self._emit(event)

    def emit_stream_cut(self, call_id: str, chunk_count: int):
        """Emit stream cut event."""
        event = Event(
            type=EventType.STREAM_CUT,
            span_id=call_id,
            data={"chunk_count": chunk_count},
        )

        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)

        self._emit(event)

    def emit_stream_stats(self, call_id: str, *, chunk_count: int):
        """Emit stream stats event (low volume)."""
        event = Event(
            type=EventType.STREAM_STATS,
            span_id=call_id,
            data={"chunk_count": chunk_count},
        )
        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)
        self._emit(event)

    def emit_token_usage(
        self,
        call_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        model: str | None = None,
    ):
        """Emit token usage event."""
        data: dict[str, Any] = {}
        if input_tokens is not None:
            data["input_tokens"] = input_tokens
        if output_tokens is not None:
            data["output_tokens"] = output_tokens
        if total_tokens is not None:
            data["total_tokens"] = total_tokens
        if model is not None:
            data["model"] = model
        event = Event(type=EventType.TOKEN_USAGE, span_id=call_id, data=data)
        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)
        self._emit(event)

    def emit_tool_use(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        input_bytes: int | None = None,
    ):
        """Emit that the LLM requested a tool (tool_use block)."""
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
        if input_bytes is not None:
            data["input_bytes"] = input_bytes
        event = Event(type=EventType.TOOL_USE, span_id=call_id, data=data)
        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)
        self._emit(event)

    def emit_tool_start(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        input_bytes: int | None = None,
        llm_args_ms: float | None = None,
    ):
        """Emit tool execution start."""
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
        if input_bytes is not None:
            data["input_bytes"] = input_bytes
        if llm_args_ms is not None:
            data["llm_args_ms"] = llm_args_ms
        event = Event(type=EventType.TOOL_START, span_id=call_id, data=data)
        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)
        self._emit(event)

    def emit_tool_end(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        success: bool,
        duration_ms: float | None = None,
        output_bytes: int | None = None,
        error: str | None = None,
        resolved_in_call_id: str | None = None,
    ):
        """Emit tool execution end."""
        data: dict[str, Any] = {"tool_name": tool_name, "success": success}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        if output_bytes is not None:
            data["output_bytes"] = output_bytes
        if error:
            data["error"] = error
        if resolved_in_call_id:
            data["resolved_in_call_id"] = resolved_in_call_id
        event = Event(type=EventType.TOOL_END, span_id=call_id, data=data)
        if call_id in self._active_spans:
            self._active_spans[call_id].events.append(event)
        self._emit(event)

    async def subscribe(self) -> asyncio.Queue[Event]:
        """Subscribe to events."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]):
        """Unsubscribe from events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def get_traces(self) -> list[Trace]:
        """Get all traces."""
        return list(self._traces.values())

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a specific trace."""
        return self._traces.get(trace_id)


event_bus = EventBus()
