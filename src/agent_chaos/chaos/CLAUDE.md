# Chaos Module

DSL for configuring chaos injection.

## Chaos Types

- **LLMChaos** (`llm.py`): API-level faults - rate limits, timeouts, malformed responses
- **ToolChaos** (`tool.py`): Tool execution faults - errors, mutations, delays
- **StreamChaos** (`stream.py`): Streaming faults - cuts, hangs, slow chunks, TTFT delays
- **HistoryChaos** (`history.py`): Context manipulation - inject/remove messages
- **UserChaos** (`user.py`): User input mutations

## Triggers

All chaos types use triggers to control when faults fire:

```python
from agent_chaos.chaos import on_turn, on_call, always, probability

LLMChaos.rate_limit(on=on_turn(0))        # First turn only
LLMChaos.timeout(on=on_call(2))           # Third LLM call
ToolChaos.error("search", on=always())    # Every time
LLMChaos.rate_limit(on=probability(0.5))  # 50% chance
```

## Builder Pattern

```python
from agent_chaos.chaos import chaos

config = (
    chaos()
    .llm.rate_limit(on=on_turn(0))
    .tool.error("get_weather")
    .stream.cut(after_chunks=10)
    .build()
)
```
