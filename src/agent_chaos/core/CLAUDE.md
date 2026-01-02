# Core Module

Runtime components for chaos testing.

## Components

- **ChaosContext** (`context.py`): Per-scenario state - holds injector, recorder, metrics, turn results
- **ChaosInjector** (`injector.py`): Evaluates chaos rules, decides when to inject faults
- **Recorder** (`recorder.py`): Emits events to sinks, delegates to MetricsStore for data
- **MetricsStore** (`metrics/`): Pydantic-based metrics collection

## MetricsStore Structure

```python
MetricsStore
├── calls: CallStats           # count, retries, by_provider, latencies
├── tokens: TokenStats         # input, output totals
├── stream: StreamStats        # ttft_times, hang_events, stream_cuts
├── tools: ToolTracking        # tool use mappings and state
├── conv: ConversationState    # entries, current_turn, system_prompt
├── history: list[CallRecord]  # completed call records
├── faults: list[FaultRecord]  # injected faults
└── _active_calls: dict        # in-flight calls (private)
```

## Usage Pattern

```python
# Recorder wraps MetricsStore
recorder = Recorder(sink=ListSink(), metrics=MetricsStore())

# ChaosContext ties everything together
ctx = ChaosContext(
    name="test",
    injector=ChaosInjector(chaos=[...]),
    recorder=recorder,
    session_id="abc",
)

# Access metrics via context
ctx.metrics.calls.count
ctx.metrics.history
```
