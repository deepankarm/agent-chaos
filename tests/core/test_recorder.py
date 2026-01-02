"""Tests for core/recorder.py - Event recorder."""

from __future__ import annotations

import pytest

from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder
from agent_chaos.events.sink import ListSink, NullSink
from agent_chaos.events.types import (
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


class TestRecorderInit:
    """Tests for Recorder initialization."""

    def test_default_init(self) -> None:
        """Recorder should work with no arguments."""
        recorder = Recorder()
        assert isinstance(recorder.sink, NullSink)
        assert recorder.metrics is None

    def test_with_sink(self) -> None:
        """Recorder should accept a sink."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        assert recorder.sink is sink

    def test_with_metrics(self) -> None:
        """Recorder should accept a MetricsStore."""
        metrics = MetricsStore()
        recorder = Recorder(metrics=metrics)
        assert recorder.metrics is metrics

    def test_with_both(self) -> None:
        """Recorder should accept both sink and metrics."""
        sink = ListSink()
        metrics = MetricsStore()
        recorder = Recorder(sink=sink, metrics=metrics)
        assert recorder.sink is sink
        assert recorder.metrics is metrics


class TestRecorderTraces:
    """Tests for trace-level recording."""

    def test_start_trace(self) -> None:
        """start_trace should emit TraceStartEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)

        trace_id = recorder.start_trace("test-scenario")

        assert trace_id != ""
        assert len(trace_id) == 8
        assert recorder.trace_id == trace_id
        assert recorder.trace_name == "test-scenario"

        assert len(sink) == 1
        event = sink.events[0]
        assert isinstance(event, TraceStartEvent)
        assert event.trace_id == trace_id
        assert event.trace_name == "test-scenario"

    def test_end_trace(self) -> None:
        """end_trace should emit TraceEndEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)

        trace_id = recorder.start_trace("test-scenario")
        recorder.end_trace(success=True)

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, TraceEndEvent)
        assert event.trace_id == trace_id
        assert event.success is True
        assert event.duration_s is not None
        assert event.duration_s >= 0

    def test_end_trace_with_error(self) -> None:
        """end_trace should record error."""
        sink = ListSink()
        recorder = Recorder(sink=sink)

        recorder.start_trace("test-scenario")
        recorder.end_trace(success=False, error="Test failed")

        event = sink.events[1]
        assert isinstance(event, TraceEndEvent)
        assert event.success is False
        assert event.error == "Test failed"

    def test_end_trace_clears_context(self) -> None:
        """end_trace should clear trace context."""
        recorder = Recorder(sink=ListSink())

        recorder.start_trace("test")
        assert recorder.trace_id != ""
        assert recorder.trace_name != ""

        recorder.end_trace()
        assert recorder.trace_id == ""
        assert recorder.trace_name == ""

    def test_end_trace_without_start(self) -> None:
        """end_trace without start should be no-op."""
        sink = ListSink()
        recorder = Recorder(sink=sink)

        recorder.end_trace()  # Should not raise
        assert len(sink) == 0


class TestRecorderSpans:
    """Tests for span-level recording."""

    def test_start_span(self) -> None:
        """start_span should emit SpanStartEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        call_id = recorder.start_span("anthropic")

        assert call_id != ""
        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, SpanStartEvent)
        assert event.span_id == call_id
        assert event.provider == "anthropic"

    def test_start_span_with_metrics(self) -> None:
        """start_span should delegate to MetricsStore."""
        metrics = MetricsStore()
        recorder = Recorder(sink=ListSink(), metrics=metrics)
        recorder.start_trace("test")

        call_id = recorder.start_span("anthropic")

        assert metrics.calls.count == 1
        assert metrics.get_active_call(call_id) is not None

    def test_end_span(self) -> None:
        """end_span should emit SpanEndEvent."""
        sink = ListSink()
        metrics = MetricsStore()
        recorder = Recorder(sink=sink, metrics=metrics)
        recorder.start_trace("test")

        call_id = recorder.start_span("anthropic")
        recorder.end_span(call_id, success=True)

        assert len(sink) == 3
        event = sink.events[2]
        assert isinstance(event, SpanEndEvent)
        assert event.span_id == call_id
        assert event.success is True
        assert event.latency_ms >= 0  # Latency calculated internally

    def test_end_span_with_error(self) -> None:
        """end_span should record error."""
        sink = ListSink()
        metrics = MetricsStore()
        recorder = Recorder(sink=sink, metrics=metrics)
        recorder.start_trace("test")

        call_id = recorder.start_span("anthropic")
        recorder.end_span(call_id, success=False, error=Exception("Rate limit"))

        event = sink.events[2]
        assert event.success is False
        assert event.error == "Rate limit"


class TestRecorderFaults:
    """Tests for fault recording."""

    def test_record_fault_basic(self) -> None:
        """record_fault should emit FaultInjectedEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_fault(
            "call1",
            "RateLimitError",
            "anthropic",
            chaos_point="LLM",
        )

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, FaultInjectedEvent)
        assert event.fault_type == "RateLimitError"
        assert event.chaos_point == "LLM"
        assert event.provider == "anthropic"

    def test_record_fault_tool(self) -> None:
        """record_fault should record tool-specific details."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_fault(
            "call1",
            "tool_error",
            chaos_point="TOOL",
            target_tool="get_weather",
            original="sunny",
            mutated="error",
        )

        event = sink.events[1]
        assert event.target_tool == "get_weather"
        assert event.original == "sunny"
        assert event.mutated == "error"

    def test_record_fault_context_mutation(self) -> None:
        """record_fault should record context mutation details."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_fault(
            "call1",
            "context_mutation",
            chaos_point="CONTEXT",
            added_messages=[{"role": "user", "content": "injected"}],
            removed_count=2,
        )

        event = sink.events[1]
        assert event.added_messages == [{"role": "user", "content": "injected"}]
        assert event.removed_count == 2

    def test_record_fault_with_metrics(self) -> None:
        """record_fault should delegate to MetricsStore."""
        metrics = MetricsStore()
        recorder = Recorder(sink=ListSink(), metrics=metrics)
        recorder.start_trace("test")

        recorder.record_fault("call1", "TestError", chaos_point="LLM")

        assert len(metrics.faults) == 1


class TestRecorderTTFT:
    """Tests for TTFT recording."""

    def test_record_ttft(self) -> None:
        """record_ttft should emit TTFTEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_ttft("call1", ttft_ms=150.0)

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, TTFTEvent)
        assert event.ttft_ms == 150.0
        assert event.is_delayed is False

    def test_record_ttft_delayed(self) -> None:
        """record_ttft should record delayed flag."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_ttft("call1", ttft_ms=5000.0, is_delayed=True)

        event = sink.events[1]
        assert event.is_delayed is True


class TestRecorderStream:
    """Tests for stream event recording."""

    def test_record_stream_cut(self) -> None:
        """record_stream_cut should emit StreamCutEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_stream_cut("call1", chunk_count=15)

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, StreamCutEvent)
        assert event.chunk_count == 15

    def test_record_stream_stats(self) -> None:
        """record_stream_stats should emit StreamStatsEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_stream_stats("call1", chunk_count=100, provider="anthropic")

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, StreamStatsEvent)
        assert event.chunk_count == 100
        assert event.provider == "anthropic"


class TestRecorderTokens:
    """Tests for token usage recording."""

    def test_record_token_usage(self) -> None:
        """record_token_usage should emit TokenUsageEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_token_usage(
            "call1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-3-opus",
        )

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, TokenUsageEvent)
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.model == "claude-3-opus"


class TestRecorderTools:
    """Tests for tool event recording."""

    def test_record_tool_use(self) -> None:
        """record_tool_use should emit ToolUseEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_tool_use(
            "call1",
            tool_name="get_weather",
            tool_use_id="tu_123",
            input_bytes=256,
            args={"city": "London"},
        )

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "get_weather"
        assert event.tool_use_id == "tu_123"
        assert event.input_bytes == 256
        assert event.args == {"city": "London"}

    def test_record_tool_start(self) -> None:
        """record_tool_start should emit ToolStartEvent."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_tool_start(
            tool_name="get_weather",
            tool_use_id="tu_123",
            call_id="call1",
            input_bytes=256,
        )

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, ToolStartEvent)
        assert event.tool_name == "get_weather"
        assert event.tool_use_id == "tu_123"

    def test_record_tool_end_success(self) -> None:
        """record_tool_end should emit ToolEndEvent for success."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_tool_end(
            tool_name="get_weather",
            success=True,
            tool_use_id="tu_123",
            call_id="call1",
            duration_ms=150.0,
            output_bytes=512,
            result='{"temp": 20}',
        )

        assert len(sink) == 2
        event = sink.events[1]
        assert isinstance(event, ToolEndEvent)
        assert event.tool_name == "get_weather"
        assert event.success is True
        assert event.duration_ms == 150.0
        assert event.result == '{"temp": 20}'

    def test_record_tool_end_error(self) -> None:
        """record_tool_end should emit ToolEndEvent for error."""
        sink = ListSink()
        recorder = Recorder(sink=sink)
        recorder.start_trace("test")

        recorder.record_tool_end(
            tool_name="get_weather",
            success=False,
            error="API timeout",
        )

        event = sink.events[1]
        assert event.success is False
        assert event.error == "API timeout"


class TestRecorderClose:
    """Tests for Recorder.close."""

    def test_close(self) -> None:
        """close should close the sink."""

        class TrackedSink:
            closed = False

            def emit(self, event):
                pass

            def close(self):
                self.closed = True

        sink = TrackedSink()
        recorder = Recorder(sink=sink)

        recorder.close()
        assert sink.closed


class TestRecorderIntegration:
    """Integration tests for Recorder."""

    def test_full_trace_with_metrics(self) -> None:
        """Test a complete trace with MetricsStore integration."""
        sink = ListSink()
        metrics = MetricsStore()
        recorder = Recorder(sink=sink, metrics=metrics)

        # Start trace
        trace_id = recorder.start_trace("integration-test")

        # Start span
        call_id = recorder.start_span("anthropic")

        # Record events
        recorder.record_ttft(call_id, ttft_ms=100.0)
        recorder.record_token_usage(
            call_id,
            input_tokens=50,
            output_tokens=25,
        )
        recorder.record_tool_use(
            call_id,
            tool_name="calculator",
            tool_use_id="tu_1",
            args={"expr": "2+2"},
        )
        recorder.record_tool_start(
            tool_name="calculator",
            tool_use_id="tu_1",
            call_id=call_id,
        )
        recorder.record_tool_end(
            tool_name="calculator",
            success=True,
            tool_use_id="tu_1",
            call_id=call_id,
            result="4",
        )

        # End span
        recorder.end_span(call_id, success=True)

        # End trace
        recorder.end_trace(success=True)

        # Verify sink received all events
        assert len(sink) == 9  # trace_start, span_start, ttft, tokens, tool_use, tool_start, tool_end, span_end, trace_end

        # Verify metrics tracked data
        assert metrics.calls.count == 1
        assert len(metrics.history) == 1
        assert metrics.history[0].success is True

        # Verify trace end has stats
        trace_end = sink.events[-1]
        assert isinstance(trace_end, TraceEndEvent)
        assert trace_end.total_calls == 1
        assert trace_end.failed_calls == 0

    def test_trace_with_faults(self) -> None:
        """Test trace with fault injection."""
        sink = ListSink()
        metrics = MetricsStore()
        recorder = Recorder(sink=sink, metrics=metrics)

        recorder.start_trace("fault-test")
        call_id = recorder.start_span("anthropic")
        recorder.record_fault(call_id, "RateLimitError", chaos_point="LLM")
        recorder.end_span(call_id, success=False, error=Exception("Rate limit exceeded"))
        recorder.end_trace(success=False, error="Chaos induced failure")

        # Check metrics tracked fault
        assert len(metrics.faults) == 1

        # Check trace end
        trace_end = sink.events[-1]
        assert trace_end.fault_count == 1
        assert trace_end.success is False
