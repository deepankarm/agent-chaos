from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_chaos.event.jsonl import JsonlEventSink
    from agent_chaos.ui.events import EventBus


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
    _tool_use_in_conversation: set[str] = field(default_factory=set)
    _event_bus: EventBus | None = field(default=None, repr=False)
    _event_sink: JsonlEventSink | None = field(default=None, repr=False)
    _trace_id: str = ""
    _trace_name: str = ""
    # Conversation tracking for UI
    conversation: list[dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.monotonic)
    _user_message_recorded: bool = False
    _current_turn: int = 0  # Current turn number (0 = no turn)

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

    def _elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        return (time.monotonic() - self._start_time) * 1000

    def set_current_turn(self, turn_number: int) -> None:
        """Set the current turn number for conversation tracking."""
        self._current_turn = turn_number

    def add_conversation_entry(
        self,
        entry_type: str,
        **kwargs: Any,
    ) -> None:
        """Add an entry to the conversation timeline."""
        # Deduplicate user messages to avoid recording the same message multiple times
        if entry_type == "user" and self._user_message_recorded:
            return

        entry: dict[str, Any] = {
            "type": entry_type,
            "timestamp_ms": self._elapsed_ms(),
        }

        # Auto-add turn_number for relevant entry types if we're in a turn
        if self._current_turn > 0 and entry_type in ("chaos", "tool_call", "tool_result"):
            entry["turn_number"] = self._current_turn

        entry.update(kwargs)
        self.conversation.append(entry)

        # Mark user message as recorded after adding it
        if entry_type == "user":
            self._user_message_recorded = True

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

    def record_fault(
        self,
        call_id: str,
        fault: Any,
        provider: str = "",
        *,
        chaos_point: str | None = None,
        chaos_fn_name: str | None = None,
        chaos_fn_doc: str | None = None,
        target_tool: str | None = None,
        original: str | None = None,
        mutated: str | None = None,
        added_messages: list[dict] | None = None,
        removed_messages: list[dict] | None = None,
        added_count: int | None = None,
        removed_count: int | None = None,
    ):
        """Record that a fault was injected.

        Args:
            chaos_point: The injection point (LLM, STREAM, TOOL, CONTEXT)
            chaos_fn_name: For custom mutations, the function name
            chaos_fn_doc: For custom mutations, the function docstring
            target_tool: For tool chaos, the affected tool name
            original: Original value before mutation
            mutated: Value after mutation
            added_messages: For context mutations, list of added messages
            removed_messages: For context mutations, list of removed messages
            added_count: Number of messages added
            removed_count: Number of messages removed
        """
        self.faults_injected.append((call_id, fault))

        # Use str(fault) if the object has a custom __str__, else fall back to class name
        fault_desc = str(fault) if hasattr(fault, "__str__") else type(fault).__name__
        # For exceptions, use the exception class name
        if isinstance(fault, Exception):
            fault_desc = type(fault).__name__

        if self._event_bus:
            self._event_bus.emit_fault(call_id, fault_desc, provider)

        if self._event_sink and self._trace_id:
            data: dict[str, Any] = {"fault_type": fault_desc}
            if chaos_point is not None:
                data["chaos_point"] = chaos_point
            if chaos_fn_name is not None:
                data["chaos_fn_name"] = chaos_fn_name
            if chaos_fn_doc is not None:
                data["chaos_fn_doc"] = chaos_fn_doc
            if target_tool is not None:
                data["target_tool"] = target_tool
            if original is not None:
                data["original"] = original
            if mutated is not None:
                data["mutated"] = mutated
            # Context mutation details
            if added_messages is not None:
                data["added_messages"] = added_messages
            if removed_messages is not None:
                data["removed_messages"] = removed_messages
            if added_count is not None:
                data["added_count"] = added_count
            if removed_count is not None:
                data["removed_count"] = removed_count

            self._event_sink.emit(
                type="fault_injected",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                data=data,
            )

        # Add to conversation timeline
        chaos_entry: dict[str, Any] = {"fault_type": fault_desc}
        if chaos_point:
            chaos_entry["chaos_point"] = chaos_point
        if chaos_fn_name:
            chaos_entry["chaos_fn_name"] = chaos_fn_name
        if chaos_fn_doc:
            chaos_entry["chaos_fn_doc"] = chaos_fn_doc
        if target_tool:
            chaos_entry["target_tool"] = target_tool
        if original:
            chaos_entry["original"] = original
        if mutated:
            chaos_entry["mutated"] = mutated
        if added_messages:
            chaos_entry["added_messages"] = added_messages
        if removed_messages:
            chaos_entry["removed_messages"] = removed_messages
        if added_count:
            chaos_entry["added_count"] = added_count
        if removed_count:
            chaos_entry["removed_count"] = removed_count
        self.add_conversation_entry("chaos", **chaos_entry)

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
        tool_args: dict[str, Any] | None = None,
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
        if tool_args is not None:
            data["args"] = tool_args

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

        # Add to conversation timeline (avoid duplicates)
        if tool_use_id and tool_use_id not in self._tool_use_in_conversation:
            self._tool_use_in_conversation.add(tool_use_id)
            self.add_conversation_entry(
                "tool_call",
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                args=tool_args,
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
        result: str | None = None,
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
        if result is not None:
            data["result"] = result
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

        # Add to conversation timeline
        self.add_conversation_entry(
            "tool_result",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            result=result,
            success=success,
            duration_ms=duration_ms,
            error=error,
        )

    def record_tool_result_seen(
        self,
        *,
        tool_use_id: str,
        is_error: bool | None = None,
        output_bytes: int | None = None,
        result: str | None = None,
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
            result=result,
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

    def record_ttft(self, ttft: float, call_id: str = "", *, is_delayed: bool = False):
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
                data={"ttft_ms": ttft * 1000, "is_delayed": is_delayed},
            )

        # Add chaos entry if this was a delayed TTFT
        if is_delayed:
            if self._event_sink and self._trace_id:
                self._event_sink.emit(
                    type="fault_injected",
                    trace_id=self._trace_id,
                    trace_name=self._trace_name,
                    span_id=call_id,
                    provider="",
                    data={
                        "fault_type": "slow_ttft",
                        "chaos_point": "STREAM",
                        "ttft_ms": ttft * 1000,
                    },
                )

            # Track as injected fault
            self.faults_injected.append((call_id, "slow_ttft"))

            self.add_conversation_entry(
                "chaos",
                fault_type="slow_ttft",
                chaos_point="STREAM",
                chaos_fn_doc=f"First token delayed by {ttft*1000:.0f}ms",
            )

    def record_hang(self, chunk_count: int, call_id: str = ""):
        """Record stream hang event."""
        self._hang_events.append(chunk_count)

        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="fault_injected",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider="",
                data={
                    "fault_type": "stream_hang",
                    "chaos_point": "STREAM",
                    "chunk_count": chunk_count,
                },
            )

        # Track as injected fault
        self.faults_injected.append((call_id, "stream_hang"))

        # Add chaos entry for stream hang
        self.add_conversation_entry(
            "chaos",
            fault_type="stream_hang",
            chaos_point="STREAM",
            chaos_fn_doc=f"Stream hung after {chunk_count} chunks",
        )

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
            # Also emit as fault_injected so it shows in chaos section
            self._event_sink.emit(
                type="fault_injected",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider="",
                data={
                    "fault_type": "stream_cut",
                    "chaos_point": "STREAM",
                    "chunk_count": chunk_count,
                },
            )

        # Track as injected fault
        self.faults_injected.append((call_id, "stream_cut"))

        # Add chaos entry for stream cut
        self.add_conversation_entry(
            "chaos",
            fault_type="stream_cut",
            chaos_point="STREAM",
            chaos_fn_doc=f"Stream terminated after {chunk_count} chunks",
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

    def record_slow_chunks(self, delay_ms: float, call_id: str = ""):
        """Record slow chunks chaos event."""
        if self._event_sink and self._trace_id:
            self._event_sink.emit(
                type="fault_injected",
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider="",
                data={
                    "fault_type": "slow_chunks",
                    "chaos_point": "STREAM",
                    "delay_ms": delay_ms,
                },
            )

        # Track as injected fault
        self.faults_injected.append((call_id, "slow_chunks"))

        # Add chaos entry for slow chunks
        self.add_conversation_entry(
            "chaos",
            fault_type="slow_chunks",
            chaos_point="STREAM",
            chaos_fn_doc=f"Each chunk delayed by {delay_ms:.0f}ms",
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
