"""MetricsStore - Central metrics collection for chaos sessions."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from agent_chaos.core.metrics.models import (
    ActiveCallInfo,
    CallRecord,
    CallStats,
    ConversationState,
    FaultRecord,
    StreamStats,
    TokenStats,
    ToolTracking,
)


class MetricsStore(BaseModel):
    """Stores metrics for a chaos session."""

    calls: CallStats = Field(default_factory=CallStats)
    tokens: TokenStats = Field(default_factory=TokenStats)
    stream: StreamStats = Field(default_factory=StreamStats)
    tools: ToolTracking = Field(default_factory=ToolTracking)
    conv: ConversationState = Field(default_factory=ConversationState)
    history: list[CallRecord] = Field(default_factory=list)
    faults: list[FaultRecord] = Field(default_factory=list)

    _active_calls: dict[str, ActiveCallInfo] = PrivateAttr(default_factory=dict)

    def get_active_call(self, call_id: str) -> ActiveCallInfo | None:
        """Get active call info by ID."""
        return self._active_calls.get(call_id)

    def get_cumulative_tokens(self) -> tuple[int, int]:
        """Get cumulative (input, output) token counts."""
        return (self.tokens.input, self.tokens.output)

    def register_tool_use(self, tool_use_id: str, tool_name: str, call_id: str) -> None:
        """Register a tool use mapping."""
        self.tools.use_to_name[tool_use_id] = tool_name
        self.tools.use_to_call_id[tool_use_id] = call_id

    def is_tool_ended(self, tool_use_id: str) -> bool:
        """Check if a tool use has ended."""
        return tool_use_id in self.tools.ended

    def mark_tool_ended(self, tool_use_id: str) -> None:
        """Mark a tool use as ended."""
        self.tools.ended.add(tool_use_id)

    def reset_user_message_flag(self) -> None:
        """Reset user message recorded flag."""
        self.conv.user_message_recorded = False

    def is_tool_in_conversation(self, tool_use_id: str) -> bool:
        """Check if tool use is already in conversation."""
        return tool_use_id in self.tools.in_conversation

    def mark_tool_in_conversation(self, tool_use_id: str) -> None:
        """Mark tool use as added to conversation."""
        self.tools.in_conversation.add(tool_use_id)

    def get_tool_name(self, tool_use_id: str) -> str:
        """Get tool name for a tool use ID."""
        return self.tools.use_to_name.get(tool_use_id, "unknown")

    def get_tool_start_time(self, tool_use_id: str) -> float | None:
        """Get start time for a tool use."""
        return self.tools.started_at.get(tool_use_id)

    def _elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        return (time.monotonic() - self.conv.start_time) * 1000

    def set_current_turn(self, turn_number: int) -> None:
        """Set the current turn number for conversation tracking."""
        self.conv.current_turn = turn_number

    def record_system_prompt(self, system_prompt: str | list[dict] | None) -> None:
        """Record the system prompt from the first LLM call. Only records once."""
        if self.conv.system_prompt_recorded or system_prompt is None:
            return

        if isinstance(system_prompt, list):
            texts = []
            for block in system_prompt:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            self.conv.system_prompt = "\n".join(texts) if texts else None
        else:
            self.conv.system_prompt = system_prompt

        self.conv.system_prompt_recorded = True

        if self.conv.system_prompt:
            system_entry: dict[str, Any] = {
                "type": "system",
                "timestamp_ms": 0,
                "content": self.conv.system_prompt,
            }
            self.conv.entries.insert(0, system_entry)

    def add_conversation_entry(self, entry_type: str, **kwargs: Any) -> None:
        """Add an entry to the conversation timeline."""
        if entry_type == "user" and self.conv.user_message_recorded:
            return

        entry: dict[str, Any] = {
            "type": entry_type,
            "timestamp_ms": self._elapsed_ms(),
        }

        if self.conv.current_turn > 0 and entry_type in ("chaos", "tool_call", "tool_result"):
            entry["turn_number"] = self.conv.current_turn

        if entry_type == "user":
            entry["cumulative_input_tokens"] = self.tokens.input
        elif entry_type == "assistant":
            entry["cumulative_output_tokens"] = self.tokens.output
            entry["cumulative_input_tokens"] = self.tokens.input

        entry.update(kwargs)
        self.conv.entries.append(entry)

        if entry_type == "user":
            self.conv.user_message_recorded = True

    def start_call(self, provider: str) -> str:
        """Start tracking a call. Returns call_id."""
        call_id = f"{provider}_{self.calls.count}_{time.monotonic()}"
        self.calls.count += 1

        if provider not in self.calls.by_provider:
            self.calls.by_provider[provider] = 0
        self.calls.by_provider[provider] += 1

        self._active_calls[call_id] = ActiveCallInfo(
            provider=provider,
            start_time=time.monotonic(),
            call_id=call_id,
        )

        return call_id

    def get_call_start_time(self, call_id: str) -> float | None:
        """Get monotonic start time for an active call."""
        info = self._active_calls.get(call_id)
        return info.start_time if info else None

    def end_call(self, call_id: str, success: bool = True, error: Exception | None = None) -> None:
        """End tracking a call."""
        if call_id not in self._active_calls:
            return

        call_info = self._active_calls.pop(call_id)
        duration = time.monotonic() - call_info.start_time

        self.history.append(
            CallRecord(
                call_id=call_id,
                provider=call_info.provider,
                success=success,
                latency=duration,
                error=str(error) if error else None,
                usage=call_info.usage,
                tool_uses=call_info.tool_uses,
                stream_chunks=call_info.stream_chunks,
            )
        )

        if success:
            self.calls.latencies.append(duration)
        elif error:
            error_str = str(error).lower()
            if any(keyword in error_str for keyword in ["rate", "timeout", "503", "429"]):
                self.calls.retries += 1

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
    ) -> None:
        """Record that a fault was injected."""
        fault_desc = str(fault) if hasattr(fault, "__str__") else type(fault).__name__
        if isinstance(fault, Exception):
            fault_desc = type(fault).__name__

        self.faults.append(FaultRecord(call_id=call_id, fault_type=fault_desc))

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
    ) -> None:
        """Record token usage for a call."""
        usage: dict[str, Any] = {}
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
            self.tokens.input += input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
            self.tokens.output += output_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        if model is not None:
            usage["model"] = model

        usage["cumulative_input_tokens"] = self.tokens.input
        usage["cumulative_output_tokens"] = self.tokens.output

        call_info = self._active_calls.get(call_id)
        if call_info:
            call_info.usage.update(usage)

    def record_tool_use(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        input_bytes: int | None = None,
        tool_args: dict[str, Any] | None = None,
        provider: str = "",
    ) -> None:
        """Record that the LLM requested a tool."""
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_use_id:
            data["tool_use_id"] = tool_use_id
            self.register_tool_use(tool_use_id, tool_name, call_id)
        if input_bytes is not None:
            data["input_bytes"] = input_bytes
        if tool_args is not None:
            data["args"] = tool_args

        call_info = self._active_calls.get(call_id)
        if call_info:
            call_info.tool_uses.append(data)

        if tool_use_id and not self.is_tool_in_conversation(tool_use_id):
            self.mark_tool_in_conversation(tool_use_id)
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
    ) -> None:
        """Record tool execution start."""
        if tool_use_id:
            self.tools.started_at.setdefault(tool_use_id, time.monotonic())

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
    ) -> None:
        """Record tool execution end."""
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
    ) -> None:
        """Non-intrusive tool execution inference."""
        if self.is_tool_ended(tool_use_id):
            return
        self.mark_tool_ended(tool_use_id)

        tool_name = self.get_tool_name(tool_use_id)
        started_at = self.get_tool_start_time(tool_use_id)
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

    def record_latency(self, call_id: str, latency: float) -> None:
        """Record latency for a call."""
        call_info = self._active_calls.get(call_id)
        if call_info:
            call_info.usage["latency"] = latency

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        if not self.calls.latencies:
            return 0.0
        return sum(self.calls.latencies) / len(self.calls.latencies)

    @property
    def total_calls(self) -> int:
        """Total number of calls."""
        return self.calls.count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        if not self.history:
            return 1.0
        successful = sum(1 for call in self.history if call.success)
        return successful / len(self.history)

    def record_ttft(self, ttft: float, call_id: str = "", *, is_delayed: bool = False) -> None:
        """Record time-to-first-token."""
        self.stream.ttft_times.append(ttft)

        if is_delayed:
            self.faults.append(FaultRecord(call_id=call_id, fault_type="slow_ttft"))
            self.add_conversation_entry(
                "chaos",
                fault_type="slow_ttft",
                chaos_point="STREAM",
                chaos_fn_doc=f"First token delayed by {ttft*1000:.0f}ms",
            )

    def record_hang(self, chunk_count: int, call_id: str = "") -> None:
        """Record stream hang event."""
        self.stream.hang_events.append(chunk_count)
        self.faults.append(FaultRecord(call_id=call_id, fault_type="stream_hang"))
        self.add_conversation_entry(
            "chaos",
            fault_type="stream_hang",
            chaos_point="STREAM",
            chaos_fn_doc=f"Stream hung after {chunk_count} chunks",
        )

    def record_stream_cut(self, chunk_count: int, call_id: str = "") -> None:
        """Record stream cut event."""
        self.stream.stream_cuts.append(chunk_count)
        self.faults.append(FaultRecord(call_id=call_id, fault_type="stream_cut"))
        self.add_conversation_entry(
            "chaos",
            fault_type="stream_cut",
            chaos_point="STREAM",
            chaos_fn_doc=f"Stream terminated after {chunk_count} chunks",
        )

    def record_stream_stats(self, call_id: str, *, chunk_count: int, provider: str = "") -> None:
        """Record final stream stats for a call."""
        call_info = self._active_calls.get(call_id)
        if call_info:
            call_info.stream_chunks = chunk_count

    def record_slow_chunks(self, delay_ms: float, call_id: str = "") -> None:
        """Record slow chunks chaos event."""
        self.faults.append(FaultRecord(call_id=call_id, fault_type="slow_chunks"))
        self.add_conversation_entry(
            "chaos",
            fault_type="slow_chunks",
            chaos_point="STREAM",
            chaos_fn_doc=f"Each chunk delayed by {delay_ms:.0f}ms",
        )

    def record_corruption(self, chunk_count: int) -> None:
        """Record corruption event."""
        self.stream.corruption_events.append(chunk_count)

    def record_chunk(self, chunk_count: int) -> None:
        """Record chunk received."""
        self.stream.chunk_counts.append(chunk_count)

    @property
    def avg_ttft(self) -> float:
        """Average time-to-first-token in seconds."""
        if not self.stream.ttft_times:
            return 0.0
        return sum(self.stream.ttft_times) / len(self.stream.ttft_times)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all completed calls."""
        total = 0
        for call in self.history:
            total += call.usage.get("input_tokens") or 0
        return total

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all completed calls."""
        total = 0
        for call in self.history:
            total += call.usage.get("output_tokens") or 0
        return total

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all completed calls."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_call(self) -> float:
        """Average total tokens per call."""
        if not self.history:
            return 0.0
        return self.total_tokens / len(self.history)

    @property
    def max_tokens_single_call(self) -> int:
        """Maximum tokens consumed in a single call."""
        max_tokens = 0
        for call in self.history:
            input_tok = call.usage.get("input_tokens") or 0
            output_tok = call.usage.get("output_tokens") or 0
            call_tokens = input_tok + output_tok
            if call_tokens > max_tokens:
                max_tokens = call_tokens
        return max_tokens

    def get_token_history(self) -> list[dict[str, Any]]:
        """Get token usage history per call for burst analysis."""
        result: list[dict[str, Any]] = []
        cumulative = 0
        for call in self.history:
            input_tok = call.usage.get("input_tokens") or 0
            output_tok = call.usage.get("output_tokens") or 0
            total = input_tok + output_tok
            cumulative += total
            result.append({
                "call_id": call.call_id,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "total_tokens": total,
                "cumulative_tokens": cumulative,
            })
        return result
