"""Metrics collection for chaos sessions."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_chaos.ui.events import EventBus
    from agent_chaos.event.jsonl import JsonlEventSink


@dataclass
class MetricsStore:
    """Stores metrics for a chaos session."""

    call_count: int = 0
    retries: int = 0
    latencies: list[float] = field(default_factory=list)
    faults_injected: list[tuple[str, Any]] = field(default_factory=list)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    _call_counts_by_provider: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _active_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    _ttft_times: list[float] = field(default_factory=list)
    _hang_events: list[int] = field(default_factory=list)
    _stream_cuts: list[int] = field(default_factory=list)
    _corruption_events: list[int] = field(default_factory=list)
    _chunk_counts: list[int] = field(default_factory=list)
    _tool_use_to_call_id: dict[str, str] = field(default_factory=dict)
    _tool_use_to_tool_name: dict[str, str] = field(default_factory=dict)
    _tool_use_started_at: dict[str, float] = field(default_factory=dict)
    _tool_use_ended: set[str] = field(default_factory=set)
    _event_bus: EventBus | None = field(default=None, repr=False)
    _event_sink: JsonlEventSink | None = field(default=None, repr=False)
    _trace_id: str = ""
    _trace_name: str = ""

    def set_event_bus(self, event_bus: EventBus):
        """Set the event bus for real-time UI updates."""
        self._event_bus = event_bus

    def set_event_sink(self, event_sink: JsonlEventSink):
        """Set a JSONL event sink for artifact persistence (CLI/CI)."""
        self._event_sink = event_sink

    def set_trace_context(self, trace_id: str, trace_name: str):
        """Set the active trace context for event sinks."""
        self._trace_id = trace_id
        self._trace_name = trace_name

    def start_call(self, provider: str) -> str:
        """Start tracking a call. Returns call_id."""
        call_id = f"{provider}_{self.call_count}_{time.monotonic()}"
        self.call_count += 1
        self._call_counts_by_provider[provider] += 1

        self._active_calls[call_id] = {
            "provider": provider,
            "start_time": time.monotonic(),
            "call_id": call_id,
            "usage": {},
            "tool_uses": [],
            "stream_chunks": 0,
        }

        if self._event_bus:
            self._event_bus.emit_call_start(call_id, provider)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="span_start",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data={},
            )

        return call_id

    def get_call_start_time(self, call_id: str) -> float | None:
        """Get monotonic start time for an active call (if still active)."""
        info = self._active_calls.get(call_id)
        if not info:
            return None
        return info.get("start_time")

    def end_call(
        self, call_id: str, success: bool = True, error: Exception | None = None
    ):
        """End tracking a call."""
        if call_id not in self._active_calls:
            return

        call_info = self._active_calls.pop(call_id)
        duration = time.monotonic() - call_info["start_time"]

        self.call_history.append(
            {
                "call_id": call_id,
                "provider": call_info["provider"],
                "success": success,
                "latency": duration,
                "error": str(error) if error else None,
                "usage": call_info.get("usage") or {},
                "tool_uses": call_info.get("tool_uses") or [],
                "stream_chunks": call_info.get("stream_chunks") or 0,
            }
        )

        if success:
            self.latencies.append(duration)
        elif error:
            # Check if this looks like a retryable error
            error_str = str(error).lower()
            if any(
                keyword in error_str for keyword in ["rate", "timeout", "503", "429"]
            ):
                self.retries += 1

        if self._event_bus:
            self._event_bus.emit_call_end(
                call_id,
                call_info["provider"],
                success,
                duration,
                str(error) if error else "",
            )

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="span_end",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=call_info["provider"],
                data={
                    "success": success,
                    "latency_ms": duration * 1000,
                    "error": str(error) if error else "",
                },
            )

    def record_fault(self, call_id: str, fault: Any, provider: str = ""):
        """Record that a fault was injected."""
        self.faults_injected.append((call_id, fault))

        if self._event_bus:
            self._event_bus.emit_fault(call_id, type(fault).__name__, provider)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="fault_injected",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data={"fault_type": type(fault).__name__},
            )

    def record_token_usage(
        self,
        call_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        model: str | None = None,
        provider: str = "",
    ):
        """Record token usage for a call (if available from provider response)."""
        usage: dict[str, Any] = {}
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        if model is not None:
            usage["model"] = model

        if call_id in self._active_calls:
            self._active_calls[call_id]["usage"] = {
                **(self._active_calls[call_id].get("usage") or {}),
                **usage,
            }

        if self._event_bus:
            self._event_bus.emit_token_usage(
                call_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
            )

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="token_usage",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data=usage,
            )

    def record_tool_use(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        input_bytes: int | None = None,
        provider: str = "",
    ):
        """Record that the LLM requested a tool (tool_use block)."""
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
            self._tool_use_to_call_id[tool_use_id] = call_id
            self._tool_use_to_tool_name[tool_use_id] = tool_name
        if input_bytes is not None:
            data["input_bytes"] = input_bytes

        if call_id in self._active_calls:
            self._active_calls[call_id].setdefault("tool_uses", []).append(data)

        if self._event_bus:
            self._event_bus.emit_tool_use(
                call_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_bytes=input_bytes,
            )

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="tool_use",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data=data,
            )

    def record_tool_start(
        self,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        call_id: str | None = None,
        input_bytes: int | None = None,
        provider: str = "",
    ):
        """Record tool execution start.

        If call_id is not provided, we try to resolve it from tool_use_id (Anthropic tool_use id).
        If we still can't resolve, this will be emitted at trace-level (span_id="").
        """
        resolved_call_id = call_id or (
            self._tool_use_to_call_id.get(tool_use_id or "") if tool_use_id else None
        )
        span_id = resolved_call_id or ""
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
            # Mark start time so we can approximate duration until tool_result is seen.
            self._tool_use_started_at.setdefault(tool_use_id, time.monotonic())
        if input_bytes is not None:
            data["input_bytes"] = input_bytes
        llm_args_ms = None
        if span_id:
            started = self.get_call_start_time(span_id)
            if started is not None:
                llm_args_ms = (time.monotonic() - started) * 1000
                data["llm_args_ms"] = llm_args_ms

        if self._event_bus:
            self._event_bus.emit_tool_start(
                span_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_bytes=input_bytes,
                llm_args_ms=llm_args_ms,
            )

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="tool_start",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=span_id,
                provider=provider,
                data=data,
            )

    def record_tool_end(
        self,
        *,
        tool_name: str,
        success: bool,
        tool_use_id: str | None = None,
        call_id: str | None = None,
        duration_ms: float | None = None,
        output_bytes: int | None = None,
        error: str | None = None,
        resolved_in_call_id: str | None = None,
        provider: str = "",
    ):
        """Record tool execution end (success/failure + duration + error)."""
        resolved_call_id = call_id or (
            self._tool_use_to_call_id.get(tool_use_id or "") if tool_use_id else None
        )
        span_id = resolved_call_id or ""
        data: dict[str, Any] = {"tool_name": tool_name, "success": success}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        if output_bytes is not None:
            data["output_bytes"] = output_bytes
        if error:
            data["error"] = error
        if resolved_in_call_id:
            data["resolved_in_call_id"] = resolved_in_call_id

        if self._event_bus:
            self._event_bus.emit_tool_end(
                span_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                success=success,
                duration_ms=duration_ms,
                output_bytes=output_bytes,
                error=error,
                resolved_in_call_id=resolved_in_call_id,
            )

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="tool_end",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=span_id,
                provider=provider,
                data=data,
            )

    def record_tool_result_seen(
        self,
        *,
        tool_use_id: str,
        is_error: bool | None = None,
        output_bytes: int | None = None,
        resolved_in_call_id: str | None = None,
        provider: str = "",
    ):
        """Non-intrusive tool execution inference.

        - tool_start: when LLM emits a tool_use block (we record start time)
        - tool_end: when we later see a tool_result block referencing that tool_use_id
        """
        if tool_use_id in self._tool_use_ended:
            return
        self._tool_use_ended.add(tool_use_id)
        tool_name = self._tool_use_to_tool_name.get(tool_use_id, "unknown")
        started_at = self._tool_use_started_at.get(tool_use_id)
        duration_ms = (time.monotonic() - started_at) * 1000 if started_at else None
        success = not bool(is_error)
        self.record_tool_end(
            tool_name=tool_name,
            success=success,
            tool_use_id=tool_use_id,
            duration_ms=duration_ms,
            output_bytes=output_bytes,
            error="tool_result.is_error=true" if is_error else None,
            resolved_in_call_id=resolved_in_call_id,
            provider=provider,
        )

    def record_latency(self, call_id: str, latency: float):
        """Record latency for a call."""
        if call_id in self._active_calls:
            self._active_calls[call_id]["latency"] = latency

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def total_calls(self) -> int:
        """Total number of calls."""
        return self.call_count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        if not self.call_history:
            return 1.0
        successful = sum(1 for call in self.call_history if call["success"])
        return successful / len(self.call_history)

    def record_ttft(self, ttft: float, call_id: str = ""):
        """Record time-to-first-token."""
        self._ttft_times.append(ttft)

        if self._event_bus:
            self._event_bus.emit_ttft(call_id, ttft)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="ttft",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider="",
                data={"ttft_ms": ttft * 1000},
            )

    def record_hang(self, chunk_count: int):
        """Record stream hang event."""
        self._hang_events.append(chunk_count)

    def record_stream_cut(self, chunk_count: int, call_id: str = ""):
        """Record stream cut event."""
        self._stream_cuts.append(chunk_count)

        if self._event_bus:
            self._event_bus.emit_stream_cut(call_id, chunk_count)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="stream_cut",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider="",
                data={"chunk_count": chunk_count},
            )

    def record_stream_stats(
        self, call_id: str, *, chunk_count: int, provider: str = ""
    ):
        """Record final stream stats for a call."""
        if call_id in self._active_calls:
            self._active_calls[call_id]["stream_chunks"] = chunk_count

        if self._event_bus:
            self._event_bus.emit_stream_stats(call_id, chunk_count=chunk_count)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="stream_stats",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data={"chunk_count": chunk_count},
            )

    def record_corruption(self, chunk_count: int):
        """Record corruption event."""
        self._corruption_events.append(chunk_count)

    def record_chunk(self, chunk_count: int):
        """Record chunk received."""
        self._chunk_counts.append(chunk_count)
        # Also keep a per-call count if we can find an active call_id (stream wrappers will
        # call record_stream_stats with the call_id at the end).

    @property
    def avg_ttft(self) -> float:
        """Average time-to-first-token in seconds."""
        if not self._ttft_times:
            return 0.0
        return sum(self._ttft_times) / len(self._ttft_times)
