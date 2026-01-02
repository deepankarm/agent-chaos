"""UI sink adapter for real-time dashboard updates.

Wraps the EventBus to implement the EventSink protocol, enabling the new
typed event system to work with the existing real-time UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_chaos.events.types import (
    Event,
    FaultInjectedEvent,
    SpanEndEvent,
    SpanStartEvent,
    StreamCutEvent,
    StreamStatsEvent,
    TokenUsageEvent,
    ToolEndEvent,
    ToolStartEvent,
    ToolUseEvent,
    TraceEndEvent,
    TraceStartEvent,
    TTFTEvent,
)

if TYPE_CHECKING:
    from agent_chaos.ui.events import EventBus


class UISink:
    """Adapter that emits typed events to the EventBus for real-time UI updates.

    This bridges the new typed event system with the existing EventBus,
    translating typed Pydantic events into EventBus method calls.

    Example:
        from agent_chaos.ui.events import event_bus
        from agent_chaos.events import MultiSink, JsonlSink

        ui_sink = UISink(event_bus)
        file_sink = JsonlSink("events.jsonl")
        sink = MultiSink([ui_sink, file_sink])

        recorder = Recorder(sink=sink, metrics=metrics)
    """

    def __init__(self, event_bus: EventBus):
        """Initialize with an EventBus instance.

        Args:
            event_bus: The EventBus to emit events to.
        """
        self._bus = event_bus

    def emit(self, event: Event) -> None:
        """Emit a typed event to the EventBus.

        Translates the typed event into the appropriate EventBus method call.
        Unknown event types are silently ignored.

        Args:
            event: The typed event to emit.
        """
        if isinstance(event, TraceStartEvent):
            # Trace start is handled separately via start_session
            pass

        elif isinstance(event, TraceEndEvent):
            # Trace end is handled separately via end_session
            pass

        elif isinstance(event, SpanStartEvent):
            self._bus.emit_call_start(event.span_id, event.provider)

        elif isinstance(event, SpanEndEvent):
            self._bus.emit_call_end(
                event.span_id,
                event.provider,
                event.success,
                event.latency_ms / 1000 if event.latency_ms else 0,
                event.error or "",
            )

        elif isinstance(event, FaultInjectedEvent):
            self._bus.emit_fault(event.span_id, event.fault_type, event.provider)

        elif isinstance(event, TTFTEvent):
            self._bus.emit_ttft(event.span_id, event.ttft_ms / 1000)

        elif isinstance(event, StreamCutEvent):
            self._bus.emit_stream_cut(event.span_id, event.chunk_count)

        elif isinstance(event, StreamStatsEvent):
            self._bus.emit_stream_stats(event.span_id, chunk_count=event.chunk_count)

        elif isinstance(event, TokenUsageEvent):
            self._bus.emit_token_usage(
                event.span_id,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                total_tokens=event.total_tokens,
                model=event.model,
            )

        elif isinstance(event, ToolUseEvent):
            self._bus.emit_tool_use(
                event.span_id,
                tool_name=event.tool_name,
                tool_use_id=event.tool_use_id,
                input_bytes=event.input_bytes,
            )

        elif isinstance(event, ToolStartEvent):
            self._bus.emit_tool_start(
                event.span_id,
                tool_name=event.tool_name,
                tool_use_id=event.tool_use_id,
                input_bytes=event.input_bytes,
                llm_args_ms=event.llm_args_ms,
            )

        elif isinstance(event, ToolEndEvent):
            self._bus.emit_tool_end(
                event.span_id,
                tool_name=event.tool_name,
                tool_use_id=event.tool_use_id,
                success=event.success,
                duration_ms=event.duration_ms,
                output_bytes=event.output_bytes,
                error=event.error,
                resolved_in_call_id=event.resolved_in_call_id,
            )

    def close(self) -> None:
        """No-op close - EventBus lifecycle is managed elsewhere."""
        pass
