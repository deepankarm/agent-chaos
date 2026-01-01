"""Tests for events/jsonl.py - JSONL event sink."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent_chaos.events.jsonl import JsonlSink, read_events
from agent_chaos.events.types import (
    FaultInjectedEvent,
    SpanEndEvent,
    SpanStartEvent,
    TraceEndEvent,
    TraceStartEvent,
    TTFTEvent,
)


class TestJsonlSink:
    """Tests for JsonlSink."""

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """JsonlSink should create parent directories."""
        path = tmp_path / "nested" / "dir" / "events.jsonl"
        sink = JsonlSink(path)
        sink.close()
        assert path.parent.exists()

    def test_emit_writes_json_line(self, tmp_path: Path) -> None:
        """JsonlSink.emit should write event as JSON line."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)

        event = TraceStartEvent(trace_id="abc", trace_name="test")
        sink.emit(event)
        sink.close()

        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["type"] == "trace_start"
        assert data["trace_id"] == "abc"
        assert data["trace_name"] == "test"

    def test_emit_multiple_events(self, tmp_path: Path) -> None:
        """JsonlSink should write multiple events as separate lines."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)

        sink.emit(TraceStartEvent(trace_id="t1"))
        sink.emit(SpanStartEvent(span_id="s1", trace_id="t1"))
        sink.emit(SpanEndEvent(span_id="s1", success=True))
        sink.emit(TraceEndEvent(trace_id="t1"))
        sink.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 4

        types = [json.loads(line)["type"] for line in lines]
        assert types == ["trace_start", "span_start", "span_end", "trace_end"]

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        """JsonlSink should append to existing file."""
        path = tmp_path / "events.jsonl"

        # Write first event
        sink1 = JsonlSink(path)
        sink1.emit(TraceStartEvent(trace_id="t1"))
        sink1.close()

        # Append second event
        sink2 = JsonlSink(path)
        sink2.emit(TraceStartEvent(trace_id="t2"))
        sink2.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["trace_id"] == "t1"
        assert json.loads(lines[1])["trace_id"] == "t2"

    def test_close_is_safe_to_call_multiple_times(self, tmp_path: Path) -> None:
        """JsonlSink.close should be safe to call multiple times."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)
        sink.emit(TraceStartEvent())
        sink.close()
        sink.close()  # Should not raise
        sink.close()  # Should not raise

    def test_context_manager(self, tmp_path: Path) -> None:
        """JsonlSink should work as context manager."""
        path = tmp_path / "events.jsonl"

        with JsonlSink(path) as sink:
            sink.emit(TraceStartEvent(trace_id="ctx"))

        # File should be closed, content should be written
        content = path.read_text()
        assert "ctx" in content

    def test_serializes_complex_events(self, tmp_path: Path) -> None:
        """JsonlSink should serialize complex event types."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)

        event = FaultInjectedEvent(
            trace_id="t1",
            span_id="s1",
            fault_type="tool_error",
            chaos_point="TOOL",
            target_tool="get_weather",
            original="sunny",
            mutated="error",
            added_messages=[{"role": "user", "content": "test"}],
            removed_count=2,
        )
        sink.emit(event)
        sink.close()

        data = json.loads(path.read_text().strip())
        assert data["fault_type"] == "tool_error"
        assert data["target_tool"] == "get_weather"
        assert data["added_messages"] == [{"role": "user", "content": "test"}]
        assert data["removed_count"] == 2


class TestReadEvents:
    """Tests for read_events function."""

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """read_events should handle empty file."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        events = read_events(path)
        assert events == []

    def test_read_single_event(self, tmp_path: Path) -> None:
        """read_events should read a single event."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)
        sink.emit(TraceStartEvent(trace_id="abc"))
        sink.close()

        events = read_events(path)
        assert len(events) == 1
        assert isinstance(events[0], TraceStartEvent)
        assert events[0].trace_id == "abc"

    def test_read_multiple_events(self, tmp_path: Path) -> None:
        """read_events should read multiple events in order."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)
        sink.emit(TraceStartEvent(trace_id="t1"))
        sink.emit(SpanStartEvent(span_id="s1"))
        sink.emit(TTFTEvent(ttft_ms=100.0))
        sink.emit(SpanEndEvent(success=True))
        sink.emit(TraceEndEvent(trace_id="t1"))
        sink.close()

        events = read_events(path)
        assert len(events) == 5
        assert isinstance(events[0], TraceStartEvent)
        assert isinstance(events[1], SpanStartEvent)
        assert isinstance(events[2], TTFTEvent)
        assert isinstance(events[3], SpanEndEvent)
        assert isinstance(events[4], TraceEndEvent)

    def test_read_preserves_data(self, tmp_path: Path) -> None:
        """read_events should preserve event data."""
        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)

        original = FaultInjectedEvent(
            trace_id="t1",
            span_id="s1",
            provider="anthropic",
            fault_type="RateLimitError",
            chaos_point="LLM",
            chaos_fn_name="custom_fn",
            target_tool="weather",
        )
        sink.emit(original)
        sink.close()

        events = read_events(path)
        loaded = events[0]

        assert isinstance(loaded, FaultInjectedEvent)
        assert loaded.trace_id == "t1"
        assert loaded.span_id == "s1"
        assert loaded.provider == "anthropic"
        assert loaded.fault_type == "RateLimitError"
        assert loaded.chaos_point == "LLM"
        assert loaded.chaos_fn_name == "custom_fn"
        assert loaded.target_tool == "weather"

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        """read_events should raise FileNotFoundError for missing file."""
        path = tmp_path / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            read_events(path)

    def test_read_ignores_blank_lines(self, tmp_path: Path) -> None:
        """read_events should skip blank lines."""
        path = tmp_path / "events.jsonl"
        path.write_text(
            '{"type": "trace_start", "trace_id": "a"}\n'
            "\n"
            '{"type": "trace_end", "trace_id": "a"}\n'
            "   \n"
        )

        events = read_events(path)
        assert len(events) == 2

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Events should roundtrip through JsonlSink and read_events."""
        path = tmp_path / "events.jsonl"

        original_events = [
            TraceStartEvent(trace_id="rt1", trace_name="roundtrip-test"),
            SpanStartEvent(span_id="s1", trace_id="rt1", provider="openai"),
            TTFTEvent(span_id="s1", ttft_ms=250.5, is_delayed=True),
            SpanEndEvent(span_id="s1", success=True, latency_ms=1500.0),
            TraceEndEvent(trace_id="rt1", total_calls=1, success=True),
        ]

        with JsonlSink(path) as sink:
            for event in original_events:
                sink.emit(event)

        loaded_events = read_events(path)

        assert len(loaded_events) == len(original_events)
        for orig, loaded in zip(original_events, loaded_events):
            assert type(orig) == type(loaded)
            assert orig.trace_id == loaded.trace_id
            if hasattr(orig, "span_id"):
                assert orig.span_id == loaded.span_id
