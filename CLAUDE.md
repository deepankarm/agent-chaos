# Agent Chaos

Chaos engineering framework for LLM agents. Injects faults into LLM API calls to test agent resilience.

## Project Structure

```
src/agent_chaos/
├── core/           # Runtime: ChaosContext, ChaosInjector, Recorder, MetricsStore
├── chaos/          # Chaos DSL (LLMChaos, ToolChaos, StreamChaos, etc.)
├── scenario/       # Test scenarios and assertions
├── patch/          # Monkey-patching for LLM providers
├── stream/         # Stream wrappers with fault injection
├── events/         # Event types and sinks for telemetry
├── integrations/   # Third-party integrations (DeepEval)
└── ui/             # Web UI for viewing test results
```

## Agent Interface

Agents are simple callables: `(ctx: ChaosContext, turn_input: str) -> dict`

```python
def my_agent(ctx, turn_input):
    # Use any LLM client - patching intercepts calls automatically
    response = client.messages.create(...)
    return {"response": response.content[0].text}
```

The framework patches LLM clients at runtime, so agents don't need special instrumentation.

## Provider Support

| Provider | Status | Notes |
|----------|--------|-------|
| Anthropic | Full | Sync/async, streaming |
| OpenAI | Stub only | Placeholder files exist |
| Gemini | Stub only | Placeholder files exist |

## Coding Conventions

- Absolute imports only (no relative imports)
- Pydantic V2 for all data models
- One-line docstrings unless complexity requires more
- No section divider comments (`# ----`)
- Tests in `tests/` mirror source structure

## CLI Usage

```bash
# Run scenarios from file, module, or directory
agent-chaos run scenarios/test_weather.py
agent-chaos run mypackage.scenarios:weather_scenario
agent-chaos run scenarios/ --recursive

# Options
agent-chaos run scenarios/ --workers 4      # Parallel execution
agent-chaos run scenarios/ --fail-fast      # Stop on first failure
agent-chaos run scenarios/ --dry-run        # List scenarios without running
agent-chaos run scenarios/ --seed 42        # Reproducible randomness

# View results in dashboard
agent-chaos ui .agent_chaos_runs
```

## Running Tests

```bash
uv run pytest tests/ -x
```

## Common Patterns

### MetricsStore Access
```python
metrics.calls.count          # Call count
metrics.tokens.input         # Cumulative input tokens
metrics.stream.ttft_times    # TTFT measurements
metrics.conv.entries         # Conversation timeline
metrics.history              # list[CallRecord]
metrics.faults               # list[FaultRecord]
```

### Chaos Configuration
```python
from agent_chaos.chaos import LLMChaos, ToolChaos, on_turn

chaos = [
    LLMChaos.rate_limit(on=on_turn(0)),
    ToolChaos.error("get_weather", on=on_turn(1)),
]
```
