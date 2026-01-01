"""Event recorder for agent-chaos.

Recorder orchestrates event emission through EventSink while delegating data
storage to MetricsStore. This provides clean separation between:
- Event emission (what gets broadcast/persisted)
- Data storage (what gets tracked for metrics/analysis)
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from agent_chaos.events.sink import EventSink, NullSink
from agent_chaos.events.types import (
    FaultInjectedEvent,
    SpanEndEvent,
    SpanStartEvent,
    StreamCutEvent,
    StreamStatsEvent,
    TokenUsageEvent,
    ToolEndEvent,
    ToolStartEvent,
    ToolUseEvent,
    TraceEndEvent,
    TraceStartEvent,
    TTFTEvent,
)

if TYPE_CHECKING:
    from agent_chaos.core.metrics import MetricsStore


class Recorder:
    """Orchestrates event emission and metrics storage.

    The Recorder is the main entry point for recording agent execution events.
    It emits typed Pydantic events through an EventSink while tracking data
    in a MetricsStore.

    Example:
        from agent_chaos.events import JsonlSink, MultiSink
        from agent_chaos.core.metrics import MetricsStore

        sink = JsonlSink("events.jsonl")
        metrics = MetricsStore()
        recorder = Recorder(sink, metrics)

        recorder.start_trace("my-scenario")
        call_id = recorder.start_span("anthropic")
        recorder.end_span(call_id, success=True)
        recorder.end_trace()
    """

    def __init__(
        self,
        sink: EventSink | None = None,
        metrics: MetricsStore | None = None,
    ):
        """Initialize the recorder.

        Args:
            sink: EventSink for emitting events. Defaults to NullSink.
            metrics: MetricsStore for data storage. Created if not provided.
        """
        self._sink: EventSink = sink if sink is not None else NullSink()
        self._metrics: MetricsStore | None = metrics
        self._trace_id: str = ""
        self._trace_name: str = ""
        self._trace_start_time: float = 0.0

    @property
    def sink(self) -> EventSink:
        """The event sink used for emission."""
        return self._sink

    @property
    def metrics(self) -> MetricsStore | None:
        """The metrics store for data storage."""
        return self._metrics

    @property
    def trace_id(self) -> str:
        """The current trace ID."""
        return self._trace_id

    @property
    def trace_name(self) -> str:
        """The current trace name."""
        return self._trace_name

    def start_trace(self, name: str, description: str = "") -> str:
        """Start a new trace (chaos session).

        Args:
            name: Name for the trace (e.g., scenario name).
            description: Optional description.

        Returns:
            The generated trace ID.
        """
        self._trace_id = str(uuid.uuid4())[:8]
        self._trace_name = name
        self._trace_start_time = time.monotonic()

        self._sink.emit(
            TraceStartEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
            )
        )

        return self._trace_id

    def end_trace(
        self,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """End the current trace.

        Args:
            success: Whether the trace completed successfully.
            error: Optional error message if failed.
        """
        if not self._trace_id:
            return

        duration_s = time.monotonic() - self._trace_start_time

        # Gather stats from metrics if available
        total_calls = 0
        failed_calls = 0
        fault_count = 0
        if self._metrics:
            total_calls = self._metrics.call_count
            failed_calls = sum(
                1 for c in self._metrics.call_history if not c.get("success", True)
            )
            fault_count = len(self._metrics.faults_injected)

        self._sink.emit(
            TraceEndEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                total_calls=total_calls,
                failed_calls=failed_calls,
                fault_count=fault_count,
                success=success,
                error=error,
                duration_s=duration_s,
            )
        )

        self._trace_id = ""
        self._trace_name = ""

    def start_call(self, provider: str) -> str:
        """Start a new LLM call.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai").

        Returns:
            The generated call ID.
        """
        call_id = ""
        if self._metrics:
            call_id = self._metrics.start_call(provider)
        else:
            # Generate a call_id if no metrics store
            call_id = f"{provider}_{time.monotonic()}"

        self._sink.emit(
            SpanStartEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
            )
        )

        return call_id

    # Alias for compatibility
    start_span = start_call

    def end_call(
        self,
        call_id: str,
        success: bool = True,
        error: Exception | None = None,
    ) -> None:
        """End an LLM call.

        Args:
            call_id: The call ID from start_call.
            success: Whether the call succeeded.
            error: Optional exception if failed.
        """
        provider = ""
        latency_ms = 0.0

        if self._metrics:
            # Get provider and latency from metrics before ending
            call_info = self._metrics._active_calls.get(call_id, {})
            provider = call_info.get("provider", "")
            start_time = call_info.get("start_time")
            if start_time:
                latency_ms = (time.monotonic() - start_time) * 1000

            self._metrics.end_call(call_id, success=success, error=error)

        error_str = str(error) if error else None
        self._sink.emit(
            SpanEndEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                success=success,
                latency_ms=latency_ms,
                error=error_str,
            )
        )

    # Alias for compatibility
    end_span = end_call

    def record_fault(
        self,
        call_id: str,
        fault_type: str,
        provider: str = "",
        *,
        chaos_point: str = "",
        chaos_fn_name: str | None = None,
        chaos_fn_doc: str | None = None,
        target_tool: str | None = None,
        original: str | None = None,
        mutated: str | None = None,
        added_messages: list[dict[str, Any]] | None = None,
        removed_messages: list[dict[str, Any]] | None = None,
        added_count: int | None = None,
        removed_count: int | None = None,
    ) -> None:
        """Record a fault injection event.

        Args:
            call_id: The span ID where fault was injected.
            fault_type: Type of fault (e.g., "RateLimitError", "stream_cut").
            provider: The LLM provider.
            chaos_point: Injection point (LLM, STREAM, TOOL, CONTEXT, USER_INPUT).
            chaos_fn_name: For custom mutations, the function name.
            chaos_fn_doc: For custom mutations, the function docstring.
            target_tool: For tool chaos, the affected tool name.
            original: Original value before mutation.
            mutated: Value after mutation.
            added_messages: For context mutations, list of added messages.
            removed_messages: For context mutations, list of removed messages.
            added_count: Number of messages added.
            removed_count: Number of messages removed.
        """
        if self._metrics:
            self._metrics.record_fault(
                call_id,
                fault_type,
                provider,
                chaos_point=chaos_point,
                chaos_fn_name=chaos_fn_name,
                chaos_fn_doc=chaos_fn_doc,
                target_tool=target_tool,
                original=original,
                mutated=mutated,
                added_messages=added_messages,
                removed_messages=removed_messages,
                added_count=added_count,
                removed_count=removed_count,
            )

        self._sink.emit(
            FaultInjectedEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                fault_type=fault_type,
                chaos_point=chaos_point,
                chaos_fn_name=chaos_fn_name,
                chaos_fn_doc=chaos_fn_doc,
                target_tool=target_tool,
                original=original,
                mutated=mutated,
                added_messages=added_messages,
                removed_messages=removed_messages,
                added_count=added_count,
                removed_count=removed_count,
            )
        )

    def record_ttft(
        self,
        call_id: str,
        ttft_ms: float,
        is_delayed: bool = False,
    ) -> None:
        """Record time-to-first-token.

        Args:
            call_id: The span ID.
            ttft_ms: Time to first token in milliseconds.
            is_delayed: Whether this TTFT was artificially delayed (chaos).
        """
        if self._metrics:
            self._metrics.record_ttft(ttft_ms / 1000, call_id, is_delayed=is_delayed)

        self._sink.emit(
            TTFTEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                ttft_ms=ttft_ms,
                is_delayed=is_delayed,
            )
        )

    def record_stream_cut(self, call_id: str, chunk_count: int) -> None:
        """Record a stream cut event.

        Args:
            call_id: The span ID.
            chunk_count: Number of chunks received before cut.
        """
        if self._metrics:
            self._metrics.record_stream_cut(chunk_count, call_id)

        self._sink.emit(
            StreamCutEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                chunk_count=chunk_count,
            )
        )

    def record_stream_stats(
        self,
        call_id: str,
        chunk_count: int,
        provider: str = "",
    ) -> None:
        """Record final stream statistics.

        Args:
            call_id: The span ID.
            chunk_count: Total number of chunks received.
            provider: The LLM provider.
        """
        if self._metrics:
            self._metrics.record_stream_stats(call_id, chunk_count=chunk_count, provider=provider)

        self._sink.emit(
            StreamStatsEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                chunk_count=chunk_count,
            )
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
    ) -> None:
        """Record token usage for a call.

        Args:
            call_id: The span ID.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            total_tokens: Total tokens (if different from sum).
            model: The model name.
            provider: The LLM provider.
        """
        cumulative_input = 0
        cumulative_output = 0
        if self._metrics:
            self._metrics.record_token_usage(
                call_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
                provider=provider,
            )
            cumulative_input = self._metrics._cumulative_input_tokens
            cumulative_output = self._metrics._cumulative_output_tokens

        self._sink.emit(
            TokenUsageEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
                cumulative_input_tokens=cumulative_input,
                cumulative_output_tokens=cumulative_output,
            )
        )

    def record_tool_use(
        self,
        call_id: str,
        *,
        tool_name: str,
        tool_use_id: str | None = None,
        input_bytes: int | None = None,
        args: dict[str, Any] | None = None,
        provider: str = "",
    ) -> None:
        """Record that the LLM requested a tool use.

        Args:
            call_id: The span ID.
            tool_name: Name of the tool.
            tool_use_id: Provider-specific tool use ID.
            input_bytes: Size of tool input in bytes.
            args: Tool arguments.
            provider: The LLM provider.
        """
        if self._metrics:
            self._metrics.record_tool_use(
                call_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_bytes=input_bytes,
                tool_args=args,
                provider=provider,
            )

        self._sink.emit(
            ToolUseEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id,
                provider=provider,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_bytes=input_bytes,
                args=args,
            )
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
        """Record tool execution start.

        Args:
            tool_name: Name of the tool.
            tool_use_id: Provider-specific tool use ID.
            call_id: The span ID (resolved from tool_use_id if not provided).
            input_bytes: Size of tool input in bytes.
            provider: The LLM provider.
        """
        llm_args_ms: float | None = None
        if self._metrics:
            self._metrics.record_tool_start(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                call_id=call_id,
                input_bytes=input_bytes,
                provider=provider,
            )
            # Try to get llm_args_ms from metrics
            if call_id:
                start_time = self._metrics.get_call_start_time(call_id)
                if start_time is not None:
                    llm_args_ms = (time.monotonic() - start_time) * 1000

        self._sink.emit(
            ToolStartEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id or "",
                provider=provider,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_bytes=input_bytes,
                llm_args_ms=llm_args_ms,
            )
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
    ) -> None:
        """Record tool execution end.

        Args:
            tool_name: Name of the tool.
            success: Whether the tool succeeded.
            tool_use_id: Provider-specific tool use ID.
            call_id: The span ID.
            duration_ms: Tool execution duration in milliseconds.
            output_bytes: Size of tool output in bytes.
            result: Tool result (truncated if large).
            error: Error message if failed.
            resolved_in_call_id: If tool was called across LLM calls.
            provider: The LLM provider.
        """
        if self._metrics:
            self._metrics.record_tool_end(
                tool_name=tool_name,
                success=success,
                tool_use_id=tool_use_id,
                call_id=call_id,
                duration_ms=duration_ms,
                output_bytes=output_bytes,
                result=result,
                error=error,
                resolved_in_call_id=resolved_in_call_id,
                provider=provider,
            )

        self._sink.emit(
            ToolEndEvent(
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                span_id=call_id or "",
                provider=provider,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                success=success,
                duration_ms=duration_ms,
                output_bytes=output_bytes,
                result=result,
                error=error,
                resolved_in_call_id=resolved_in_call_id,
            )
        )

    def record_latency(self, call_id: str, latency: float) -> None:
        """Record latency for a call (data only, no event).

        Args:
            call_id: The call ID.
            latency: Latency in seconds.
        """
        if self._metrics:
            self._metrics.record_latency(call_id, latency)

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
        """Infer tool completion from seeing tool_result in messages.

        This is called when we see a tool_result block, allowing us to
        infer that a tool completed even if we didn't see the execution.

        Args:
            tool_use_id: The tool use ID being resolved.
            is_error: Whether the tool result indicates an error.
            output_bytes: Size of tool output in bytes.
            result: The tool result content.
            resolved_in_call_id: The call ID where this result was seen.
            provider: The LLM provider.
        """
        if not self._metrics:
            return

        # Check if already processed
        if tool_use_id in self._metrics._tool_use_ended:
            return

        self._metrics._tool_use_ended.add(tool_use_id)

        tool_name = self._metrics._tool_use_to_tool_name.get(tool_use_id, "unknown")
        started_at = self._metrics._tool_use_started_at.get(tool_use_id)
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

    def add_conversation_entry(
        self,
        entry_type: str,
        **kwargs: Any,
    ) -> None:
        """Add an entry to the conversation history.

        This is a data-only operation (no event emitted).

        Args:
            entry_type: Type of entry (e.g., "user", "assistant", "tool_call").
            **kwargs: Additional entry data.
        """
        if self._metrics:
            self._metrics.add_conversation_entry(entry_type, **kwargs)

    def record_system_prompt(self, system: str | list[dict[str, Any]]) -> None:
        """Record the system prompt (data-only, no event emitted).

        Args:
            system: The system prompt content.
        """
        if self._metrics:
            self._metrics.record_system_prompt(system)

    def close(self) -> None:
        """Close the recorder and its sink."""
        self._sink.close()
