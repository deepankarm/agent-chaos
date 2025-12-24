"""JSONL event sink for agent-chaos.

Writes events compatible with the dashboard event schema:
{type, timestamp, trace_id, trace_name, span_id, provider, data}.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceContext:
    trace_id: str
    trace_name: str


class JsonlEventSink:
    """Append-only JSONL event sink.

    This is meant for CLI/CI artifacts (replay + postmortems).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def start_trace(self, name: str) -> TraceContext:
        trace_id = str(uuid.uuid4())[:8]
        ctx = TraceContext(trace_id=trace_id, trace_name=name)
        self.emit(
            type="trace_start",
            trace_id=ctx.trace_id,
            trace_name=ctx.trace_name,
            span_id="",
            provider="",
            data={},
        )
        return ctx

    def end_trace(self, trace_id: str, trace_name: str, data: dict[str, Any]) -> None:
        self.emit(
            type="trace_end",
            trace_id=trace_id,
            trace_name=trace_name,
            span_id="",
            provider="",
            data=data,
        )

    def emit(
        self,
        *,
        type: str,
        trace_id: str,
        trace_name: str,
        span_id: str = "",
        provider: str = "",
        data: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        event = {
            "type": type,
            "timestamp": timestamp or _now_iso(),
            "trace_id": trace_id,
            "trace_name": trace_name,
            "span_id": span_id,
            "provider": provider,
            "data": data or {},
        }
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()
