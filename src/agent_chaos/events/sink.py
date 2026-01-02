"""Event sink protocol and implementations for agent-chaos.

EventSink provides a unified interface for event emission. Sinks can write to
JSONL files, broadcast to real-time UI, or both via MultiSink.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_chaos.events.types import Event


@runtime_checkable
class EventSink(Protocol):
    """Protocol for event sinks.

    Any class implementing this protocol can receive events from the system.
    Examples include JSONL file writers, real-time UI broadcasters, or test mocks.
    """

    def emit(self, event: Event) -> None:
        """Emit an event to this sink.

        Args:
            event: The event to emit. Must be a subclass of BaseEvent.
        """
        ...

    def close(self) -> None:
        """Close the sink and release any resources.

        Called when the sink is no longer needed. Implementations should
        ensure all buffered events are flushed before returning.
        """
        ...


class MultiSink:
    """Composite sink that broadcasts events to multiple sinks.

    This allows emitting to both JSONL (for persistence) and UI (for real-time
    display) with a single call.

    Example:
        jsonl = JsonlSink("events.jsonl")
        ui = UISink()
        sink = MultiSink([jsonl, ui])
        sink.emit(SpanStartEvent(span_id="abc"))  # Goes to both
    """

    def __init__(self, sinks: list[EventSink] | None = None):
        """Initialize with a list of sinks.

        Args:
            sinks: List of sinks to broadcast to. Can be empty.
        """
        self._sinks: list[EventSink] = list(sinks) if sinks else []

    def add(self, sink: EventSink) -> None:
        """Add a sink to the broadcast list.

        Args:
            sink: The sink to add.
        """
        self._sinks.append(sink)

    def remove(self, sink: EventSink) -> None:
        """Remove a sink from the broadcast list.

        Args:
            sink: The sink to remove. No error if not present.
        """
        try:
            self._sinks.remove(sink)
        except ValueError:
            pass

    def emit(self, event: Event) -> None:
        """Emit an event to all registered sinks.

        Errors in individual sinks are caught and ignored to prevent
        one failing sink from breaking others.

        Args:
            event: The event to emit.
        """
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                # Don't let one sink's failure break others
                pass

    def close(self) -> None:
        """Close all registered sinks.

        Errors during close are caught and ignored.
        """
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass

    def __len__(self) -> int:
        """Return the number of registered sinks."""
        return len(self._sinks)


class NullSink:
    """A sink that discards all events.

    Useful for testing or when events are not needed.
    """

    def emit(self, event: Event) -> None:
        """Discard the event."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass


class ListSink:
    """A sink that collects events into a list.

    Useful for testing and inspection.

    Example:
        sink = ListSink()
        recorder.emit(SpanStartEvent(...))
        assert len(sink.events) == 1
    """

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        """Append the event to the internal list."""
        self.events.append(event)

    def close(self) -> None:
        """No-op close."""
        pass

    def clear(self) -> None:
        """Clear all collected events."""
        self.events.clear()

    def __len__(self) -> int:
        """Return the number of collected events."""
        return len(self.events)
