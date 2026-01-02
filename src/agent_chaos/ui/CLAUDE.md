# UI Module

FastAPI-based web dashboard for viewing chaos test results.

## Components

- `server.py` - FastAPI app with uvicorn server
- `events.py` - EventBus for real-time WebSocket updates
- `static/` - HTML, CSS, JavaScript assets

## Usage

```bash
# CLI
agent-chaos ui --port 8765

# Programmatically
from agent_chaos.ui.server import run_server
run_server(runs_dir=Path(".agent_chaos_runs"), port=8765)
```

## API Endpoints

- `GET /` - Dashboard HTML
- `GET /api/traces` - All traces (live + artifacts)
- `WS /ws` - Real-time event stream

## Event Broadcasting

The UI receives events via `UISink` bridging to `EventBus`:

```python
from agent_chaos.events.ui_sink import UISink
from agent_chaos.events import MultiSink
from agent_chaos.events.jsonl import JsonlSink
from agent_chaos.ui.events import event_bus

sink = MultiSink([
    JsonlSink("events.jsonl"),
    UISink(event_bus),
])
```

## Artifact Loading

Dashboard loads historical runs from `runs_dir` by parsing:
- `scorecard.json` - Scenario results
- `events.jsonl` - Event timeline
