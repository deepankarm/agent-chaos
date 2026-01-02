"""JSONL event sink for agent-chaos.

Writes Pydantic event models to JSONL files for persistence and replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

from agent_chaos.events.types import Event


class JsonlSink:
    """Append-only JSONL event sink.

    Writes Pydantic events to a JSONL file. Each line is a complete JSON object
    that can be deserialized back to the original event type using the
    discriminated union.

    This is meant for CLI/CI artifacts (replay + postmortems).

    Example:
        sink = JsonlSink("events.jsonl")
        sink.emit(SpanStartEvent(trace_id="abc", span_id="123"))
        sink.close()
    """

    def __init__(self, path: str | Path):
        """Initialize the JSONL sink.

        Args:
            path: Path to the JSONL file. Parent directories will be created.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = self.path.open("a", encoding="utf-8")

    def emit(self, event: Event) -> None:
        """Write an event to the JSONL file.

        The event is serialized using Pydantic's model_dump_json which
        handles datetime serialization and other type conversions.

        Args:
            event: The event to write.
        """
        # model_dump_json returns a JSON string without newline
        self._fh.write(event.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file handle.

        Safe to call multiple times.
        """
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> JsonlSink:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


def read_events(path: str | Path) -> list[Event]:
    """Read events from a JSONL file.

    Uses Pydantic's discriminated union to deserialize each line to the
    correct event type based on the 'type' field.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of Event objects in file order.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValidationError: If any line fails to parse.
    """
    from pydantic import TypeAdapter

    adapter = TypeAdapter(Event)
    events: list[Event] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(adapter.validate_json(line))

    return events
