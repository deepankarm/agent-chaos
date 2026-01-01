"""Tests for events/types.py - Pydantic event models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter

from agent_chaos.events.types import (
    BaseEvent,
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


class TestBaseEvent:
    """Tests for BaseEvent base class."""

    def test_default_timestamp(self) -> None:
        """BaseEvent should have a default timestamp."""
        event = TraceStartEvent()
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == timezone.utc

    def test_default_empty_strings(self) -> None:
        """BaseEvent should have empty default strings."""
        event = TraceStartEvent()
        assert event.trace_id == ""
        assert event.trace_name == ""
        assert event.span_id == ""
        assert event.provider == ""

    def test_extra_fields_allowed(self) -> None:
        """BaseEvent should allow extra fields."""
        event = TraceStartEvent(trace_id="abc", custom_field="value")
        assert event.trace_id == "abc"
        assert event.model_extra.get("custom_field") == "value"


class TestTraceEvents:
    """Tests for trace-level events."""

    def test_trace_start_event(self) -> None:
        event = TraceStartEvent(
            trace_id="abc123",
            trace_name="test-scenario",
        )
        assert event.type == "trace_start"
        assert event.trace_id == "abc123"
        assert event.trace_name == "test-scenario"

    def test_trace_end_event(self) -> None:
        event = TraceEndEvent(
            trace_id="abc123",
            trace_name="test-scenario",
            total_calls=5,
            failed_calls=1,
            fault_count=2,
            success=False,
            error="Test error",
            duration_s=10.5,
        )
        assert event.type == "trace_end"
        assert event.total_calls == 5
        assert event.failed_calls == 1
        assert event.fault_count == 2
        assert event.success is False
        assert event.error == "Test error"
        assert event.duration_s == 10.5

    def test_trace_end_defaults(self) -> None:
        event = TraceEndEvent()
        assert event.total_calls == 0
        assert event.failed_calls == 0
        assert event.fault_count == 0
        assert event.success is True
        assert event.error is None
        assert event.duration_s is None


class TestSpanEvents:
    """Tests for span-level events."""

    def test_span_start_event(self) -> None:
        event = SpanStartEvent(
            trace_id="abc",
            span_id="span1",
            provider="anthropic",
        )
        assert event.type == "span_start"
        assert event.span_id == "span1"
        assert event.provider == "anthropic"

    def test_span_end_event(self) -> None:
        event = SpanEndEvent(
            span_id="span1",
            success=True,
            latency_ms=150.5,
        )
        assert event.type == "span_end"
        assert event.success is True
        assert event.latency_ms == 150.5
        assert event.error is None

    def test_span_end_with_error(self) -> None:
        event = SpanEndEvent(
            span_id="span1",
            success=False,
            latency_ms=50.0,
            error="Rate limit exceeded",
        )
        assert event.success is False
        assert event.error == "Rate limit exceeded"


class TestFaultInjectedEvent:
    """Tests for FaultInjectedEvent."""

    def test_basic_fault(self) -> None:
        event = FaultInjectedEvent(
            span_id="span1",
            fault_type="RateLimitError",
            chaos_point="LLM",
        )
        assert event.type == "fault_injected"
        assert event.fault_type == "RateLimitError"
        assert event.chaos_point == "LLM"

    def test_tool_fault(self) -> None:
        event = FaultInjectedEvent(
            fault_type="tool_error",
            chaos_point="TOOL",
            target_tool="get_weather",
            original="sunny",
            mutated="Service unavailable",
        )
        assert event.target_tool == "get_weather"
        assert event.original == "sunny"
        assert event.mutated == "Service unavailable"

    def test_context_mutation(self) -> None:
        event = FaultInjectedEvent(
            fault_type="context_mutation",
            chaos_point="CONTEXT",
            added_messages=[{"role": "user", "content": "Ignore previous"}],
            removed_messages=[{"role": "assistant", "content": "Hello"}],
            added_count=1,
            removed_count=1,
        )
        assert event.added_messages == [{"role": "user", "content": "Ignore previous"}]
        assert event.removed_messages == [{"role": "assistant", "content": "Hello"}]
        assert event.added_count == 1
        assert event.removed_count == 1

    def test_custom_chaos_function(self) -> None:
        event = FaultInjectedEvent(
            fault_type="custom",
            chaos_point="TOOL",
            chaos_fn_name="corrupt_json",
            chaos_fn_doc="Corrupts JSON responses randomly",
        )
        assert event.chaos_fn_name == "corrupt_json"
        assert event.chaos_fn_doc == "Corrupts JSON responses randomly"


class TestTTFTEvent:
    """Tests for TTFTEvent."""

    def test_basic_ttft(self) -> None:
        event = TTFTEvent(
            span_id="span1",
            ttft_ms=150.0,
        )
        assert event.type == "ttft"
        assert event.ttft_ms == 150.0
        assert event.is_delayed is False

    def test_delayed_ttft(self) -> None:
        event = TTFTEvent(
            span_id="span1",
            ttft_ms=5000.0,
            is_delayed=True,
        )
        assert event.is_delayed is True


class TestStreamEvents:
    """Tests for stream-related events."""

    def test_stream_cut_event(self) -> None:
        event = StreamCutEvent(
            span_id="span1",
            chunk_count=15,
        )
        assert event.type == "stream_cut"
        assert event.chunk_count == 15

    def test_stream_stats_event(self) -> None:
        event = StreamStatsEvent(
            span_id="span1",
            chunk_count=100,
        )
        assert event.type == "stream_stats"
        assert event.chunk_count == 100


class TestTokenUsageEvent:
    """Tests for TokenUsageEvent."""

    def test_full_usage(self) -> None:
        event = TokenUsageEvent(
            span_id="span1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-3-opus-20240229",
            cumulative_input_tokens=500,
            cumulative_output_tokens=250,
        )
        assert event.type == "token_usage"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.model == "claude-3-opus-20240229"
        assert event.cumulative_input_tokens == 500
        assert event.cumulative_output_tokens == 250

    def test_partial_usage(self) -> None:
        event = TokenUsageEvent(
            span_id="span1",
            output_tokens=50,
        )
        assert event.input_tokens is None
        assert event.output_tokens == 50
        assert event.total_tokens is None


class TestToolEvents:
    """Tests for tool-related events."""

    def test_tool_use_event(self) -> None:
        event = ToolUseEvent(
            span_id="span1",
            tool_name="get_weather",
            tool_use_id="tu_123",
            input_bytes=256,
            args={"city": "London"},
        )
        assert event.type == "tool_use"
        assert event.tool_name == "get_weather"
        assert event.tool_use_id == "tu_123"
        assert event.input_bytes == 256
        assert event.args == {"city": "London"}

    def test_tool_start_event(self) -> None:
        event = ToolStartEvent(
            span_id="span1",
            tool_name="get_weather",
            tool_use_id="tu_123",
            input_bytes=256,
            llm_args_ms=50.0,
        )
        assert event.type == "tool_start"
        assert event.tool_name == "get_weather"
        assert event.llm_args_ms == 50.0

    def test_tool_end_success(self) -> None:
        event = ToolEndEvent(
            span_id="span1",
            tool_name="get_weather",
            tool_use_id="tu_123",
            success=True,
            duration_ms=150.0,
            output_bytes=512,
            result='{"temp": 20}',
        )
        assert event.type == "tool_end"
        assert event.success is True
        assert event.duration_ms == 150.0
        assert event.result == '{"temp": 20}'
        assert event.error is None

    def test_tool_end_error(self) -> None:
        event = ToolEndEvent(
            span_id="span1",
            tool_name="get_weather",
            success=False,
            error="API timeout",
        )
        assert event.success is False
        assert event.error == "API timeout"

    def test_tool_end_resolved_in_call(self) -> None:
        event = ToolEndEvent(
            span_id="span2",
            tool_name="get_weather",
            tool_use_id="tu_123",
            success=True,
            resolved_in_call_id="call_456",
        )
        assert event.resolved_in_call_id == "call_456"


class TestEventUnion:
    """Tests for the Event discriminated union."""

    def test_discriminator(self) -> None:
        """Event union should use 'type' as discriminator."""
        adapter = TypeAdapter(Event)

        # Parse different event types
        trace_start = adapter.validate_python({"type": "trace_start"})
        assert isinstance(trace_start, TraceStartEvent)

        span_end = adapter.validate_python(
            {"type": "span_end", "success": True, "latency_ms": 100.0}
        )
        assert isinstance(span_end, SpanEndEvent)

        fault = adapter.validate_python(
            {"type": "fault_injected", "fault_type": "RateLimitError"}
        )
        assert isinstance(fault, FaultInjectedEvent)

    def test_json_serialization(self) -> None:
        """Events should serialize to JSON correctly."""
        event = SpanStartEvent(
            trace_id="abc",
            span_id="span1",
            provider="anthropic",
        )
        json_str = event.model_dump_json()
        assert '"type":"span_start"' in json_str
        assert '"trace_id":"abc"' in json_str

    def test_json_deserialization(self) -> None:
        """Events should deserialize from JSON correctly."""
        adapter = TypeAdapter(Event)
        json_str = '{"type": "ttft", "span_id": "s1", "ttft_ms": 100.0}'
        event = adapter.validate_json(json_str)
        assert isinstance(event, TTFTEvent)
        assert event.ttft_ms == 100.0
