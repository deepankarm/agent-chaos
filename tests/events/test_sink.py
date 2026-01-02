"""Tests for events/sink.py - EventSink protocol and implementations."""

from __future__ import annotations

import pytest

from agent_chaos.events.sink import EventSink, ListSink, MultiSink, NullSink
from agent_chaos.events.types import SpanEndEvent, SpanStartEvent, TraceStartEvent


class TestEventSinkProtocol:
    """Tests for EventSink protocol."""

    def test_list_sink_is_event_sink(self) -> None:
        """ListSink should satisfy EventSink protocol."""
        sink = ListSink()
        assert isinstance(sink, EventSink)

    def test_null_sink_is_event_sink(self) -> None:
        """NullSink should satisfy EventSink protocol."""
        sink = NullSink()
        assert isinstance(sink, EventSink)

    def test_multi_sink_is_event_sink(self) -> None:
        """MultiSink should satisfy EventSink protocol."""
        sink = MultiSink()
        assert isinstance(sink, EventSink)


class TestNullSink:
    """Tests for NullSink."""

    def test_emit_does_nothing(self) -> None:
        """NullSink.emit should do nothing."""
        sink = NullSink()
        event = TraceStartEvent(trace_id="abc")
        # Should not raise
        sink.emit(event)

    def test_close_does_nothing(self) -> None:
        """NullSink.close should do nothing."""
        sink = NullSink()
        # Should not raise
        sink.close()


class TestListSink:
    """Tests for ListSink."""

    def test_emit_collects_events(self) -> None:
        """ListSink should collect emitted events."""
        sink = ListSink()
        event1 = TraceStartEvent(trace_id="abc")
        event2 = SpanStartEvent(span_id="span1")

        sink.emit(event1)
        sink.emit(event2)

        assert len(sink.events) == 2
        assert sink.events[0] == event1
        assert sink.events[1] == event2

    def test_len(self) -> None:
        """ListSink should support len()."""
        sink = ListSink()
        assert len(sink) == 0

        sink.emit(TraceStartEvent())
        assert len(sink) == 1

        sink.emit(SpanStartEvent())
        assert len(sink) == 2

    def test_clear(self) -> None:
        """ListSink.clear should remove all events."""
        sink = ListSink()
        sink.emit(TraceStartEvent())
        sink.emit(SpanStartEvent())
        assert len(sink) == 2

        sink.clear()
        assert len(sink) == 0
        assert sink.events == []

    def test_close_does_nothing(self) -> None:
        """ListSink.close should not affect events."""
        sink = ListSink()
        sink.emit(TraceStartEvent())
        sink.close()
        assert len(sink) == 1


class TestMultiSink:
    """Tests for MultiSink."""

    def test_empty_multi_sink(self) -> None:
        """MultiSink with no sinks should work."""
        sink = MultiSink()
        event = TraceStartEvent()
        # Should not raise
        sink.emit(event)
        sink.close()

    def test_init_with_sinks(self) -> None:
        """MultiSink should accept sinks in constructor."""
        list1 = ListSink()
        list2 = ListSink()
        multi = MultiSink([list1, list2])
        assert len(multi) == 2

    def test_broadcast_to_all(self) -> None:
        """MultiSink should broadcast to all sinks."""
        list1 = ListSink()
        list2 = ListSink()
        multi = MultiSink([list1, list2])

        event = TraceStartEvent(trace_id="abc")
        multi.emit(event)

        assert len(list1) == 1
        assert len(list2) == 1
        assert list1.events[0] == event
        assert list2.events[0] == event

    def test_add_sink(self) -> None:
        """MultiSink.add should add a sink."""
        multi = MultiSink()
        assert len(multi) == 0

        list1 = ListSink()
        multi.add(list1)
        assert len(multi) == 1

        event = TraceStartEvent()
        multi.emit(event)
        assert len(list1) == 1

    def test_remove_sink(self) -> None:
        """MultiSink.remove should remove a sink."""
        list1 = ListSink()
        list2 = ListSink()
        multi = MultiSink([list1, list2])

        multi.remove(list1)
        assert len(multi) == 1

        event = TraceStartEvent()
        multi.emit(event)
        assert len(list1) == 0  # Not receiving
        assert len(list2) == 1  # Still receiving

    def test_remove_nonexistent_sink(self) -> None:
        """MultiSink.remove with nonexistent sink should not raise."""
        multi = MultiSink()
        other = ListSink()
        # Should not raise
        multi.remove(other)

    def test_close_all(self) -> None:
        """MultiSink.close should close all sinks."""

        class TrackedSink:
            closed = False

            def emit(self, event):
                pass

            def close(self):
                self.closed = True

        sink1 = TrackedSink()
        sink2 = TrackedSink()
        multi = MultiSink([sink1, sink2])

        multi.close()
        assert sink1.closed
        assert sink2.closed

    def test_error_isolation(self) -> None:
        """Errors in one sink should not affect others."""

        class FailingSink:
            def emit(self, event):
                raise RuntimeError("Intentional failure")

            def close(self):
                pass

        list_sink = ListSink()
        failing_sink = FailingSink()
        multi = MultiSink([failing_sink, list_sink])

        event = TraceStartEvent()
        # Should not raise despite FailingSink
        multi.emit(event)
        # list_sink should still receive the event
        assert len(list_sink) == 1

    def test_close_error_isolation(self) -> None:
        """Errors in close should not affect other sinks."""

        class FailingCloseSink:
            closed = False

            def emit(self, event):
                pass

            def close(self):
                raise RuntimeError("Close failed")

        class TrackedSink:
            closed = False

            def emit(self, event):
                pass

            def close(self):
                self.closed = True

        failing = FailingCloseSink()
        tracked = TrackedSink()
        multi = MultiSink([failing, tracked])

        # Should not raise
        multi.close()
        # tracked should still be closed
        assert tracked.closed

    def test_len(self) -> None:
        """MultiSink should support len()."""
        multi = MultiSink()
        assert len(multi) == 0

        multi.add(ListSink())
        assert len(multi) == 1

        multi.add(NullSink())
        assert len(multi) == 2


class TestMultipleSinkTypes:
    """Tests for using different sink types together."""

    def test_mixed_sinks(self) -> None:
        """MultiSink should work with mixed sink types."""
        list_sink = ListSink()
        null_sink = NullSink()
        multi = MultiSink([list_sink, null_sink])

        events = [
            TraceStartEvent(trace_id="t1"),
            SpanStartEvent(span_id="s1"),
            SpanEndEvent(span_id="s1", success=True),
        ]

        for event in events:
            multi.emit(event)

        assert len(list_sink) == 3
        # null_sink discards everything, which is fine
