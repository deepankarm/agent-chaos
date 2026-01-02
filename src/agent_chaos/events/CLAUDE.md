# Events Module

Pydantic event types and sinks for telemetry.

## Event Types (`types.py`)

All events inherit from `BaseEvent` with timestamp, trace_id, span_id, provider:

- `TraceStartEvent` / `TraceEndEvent` - Scenario boundaries
- `SpanStartEvent` / `SpanEndEvent` - LLM call boundaries
- `FaultInjectedEvent` - Chaos injection occurred
- `TTFTEvent` - Time-to-first-token recorded
- `StreamCutEvent` / `StreamStatsEvent` - Stream events
- `TokenUsageEvent` - Token consumption
- `ToolUseEvent` / `ToolStartEvent` / `ToolEndEvent` - Tool lifecycle

## Sinks

### Protocol (`sink.py`)
- `EventSink` - Protocol with `emit(event)` and `close()` methods

### Implementations
- `MultiSink` - Broadcasts to multiple sinks
- `NullSink` - Discards all events (for testing)
- `ListSink` - Collects events in memory (for testing)
- `JsonlSink` (`jsonl.py`) - Writes to JSONL file
- `UISink` (`ui_sink.py`) - Bridges to UI EventBus

## Usage

```python
from agent_chaos.events import ListSink, MultiSink
from agent_chaos.events.jsonl import JsonlSink

# Single sink
sink = JsonlSink("events.jsonl")

# Multiple sinks
sink = MultiSink([JsonlSink("events.jsonl"), ListSink()])

# After scenario
for event in list_sink.events:
    print(event)
```
