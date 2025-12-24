"""Event sinks and persistence for agent-chaos.

This package is intentionally tiny for feasibility:
- JSONL event sink for replay/debugging
- shared event schema aligned with the dashboard (`agent_chaos.ui.events`)
"""

from agent_chaos.event.jsonl import JsonlEventSink

__all__ = ["JsonlEventSink"]
