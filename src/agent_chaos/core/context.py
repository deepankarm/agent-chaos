"""Chaos context manager and context object."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore

if TYPE_CHECKING:
    from agent_chaos.scenario.model import TurnResult


class ChaosContext:
    """Context object providing access to injector and metrics."""

    def __init__(
        self,
        name: str,
        injector: ChaosInjector,
        metrics: MetricsStore,
        session_id: str,
    ):
        self.name = name
        self.injector = injector
        self.metrics = metrics
        self.session_id = session_id
        self.result: Any | None = None
        self.error: str | None = None
        self.elapsed_s: float | None = None
        self.agent_input: str | None = None
        self.agent_output: str | None = None

        # Turn tracking
        self.current_turn: int = 0
        self.turn_results: list[TurnResult] = []
        self._turn_start_calls: int = 0  # LLM calls at turn start
        self._turn_start_time: float = 0.0

        # Agent state - persists across turns for framework-specific data
        # (e.g., pydantic-ai message_history, langchain memory, etc.)
        self.agent_state: dict[str, Any] = {}

    def start_turn(self, turn_number: int, turn_input: str) -> None:
        """Called by framework at start of each turn.

        Args:
            turn_number: 1-indexed turn number.
            turn_input: The input text for this turn.
        """
        import time

        self.current_turn = turn_number
        self._turn_start_calls = self.metrics.total_calls
        self._turn_start_time = time.monotonic()

        # Reset user message flag so each turn can record its user message
        self.metrics._user_message_recorded = False

        # Update metrics current turn for conversation tracking
        self.metrics.set_current_turn(turn_number)

        # Update injector's current turn for chaos triggering
        self.injector.set_current_turn(turn_number)

        # Record turn start in conversation
        self.metrics.add_conversation_entry(
            "turn_start",
            turn_number=turn_number,
            input=turn_input,
            input_type="dynamic" if hasattr(self, "_current_turn_dynamic") and self._current_turn_dynamic else "static",
        )

    def end_turn(
        self,
        turn_input: str,
        response: str,
        success: bool,
        error: str | None = None,
    ) -> "TurnResult":
        """Called by framework at end of each turn.

        Args:
            turn_input: The input text for this turn.
            response: The agent's response.
            success: Whether the turn completed successfully.
            error: Error message if turn failed.

        Returns:
            TurnResult for this turn.
        """
        import time

        from agent_chaos.scenario.model import TurnResult

        duration_s = time.monotonic() - self._turn_start_time
        llm_calls = self.metrics.total_calls - self._turn_start_calls

        turn_result = TurnResult(
            turn_number=self.current_turn,
            input=turn_input,
            response=response,
            success=success,
            duration_s=duration_s,
            llm_calls=llm_calls,
            error=error,
        )
        self.turn_results.append(turn_result)

        # Record turn end in conversation
        self.metrics.add_conversation_entry(
            "turn_end",
            turn_number=self.current_turn,
            success=success,
            duration_s=duration_s,
            llm_calls=llm_calls,
            error=error,
        )

        return turn_result

    def get_turn_result(self, turn_number: int) -> "TurnResult | None":
        """Get the result for a specific turn.

        Args:
            turn_number: 1-indexed turn number.

        Returns:
            TurnResult for the turn, or None if not found.
        """
        for result in self.turn_results:
            if result.turn_number == turn_number:
                return result
        return None


@contextmanager
def chaos_context(
    name: str,
    chaos: list[Chaos | ChaosBuilder] | None = None,
    providers: list[str] | None = None,
    emit_events: bool = False,
    event_sink: Any | None = None,
    description: str = "",
) -> Iterator[ChaosContext]:
    """Context manager for scoped chaos injection.

    Introduce a little chaos at every boundary of your agent.

    Args:
        name: Name for this chaos context (shown in UI)
        chaos: List of chaos to inject
        providers: List of providers to patch (default: ["anthropic"])
        emit_events: If True, emit events to the UI dashboard
        event_sink: Optional event sink for artifact persistence (e.g. JSONL)
        description: Optional description of the scenario (shown in UI)

    Yields:
        ChaosContext with injector and metrics access

    Example:
        from agent_chaos import (
            chaos_context,
            llm_rate_limit,
            llm_stream_cut,
            tool_error,
        )

        with chaos_context(
            name="test",
            description="Tests agent resilience to various failures",
            chaos=[
                llm_rate_limit().after_calls(2),
                llm_stream_cut(after_chunks=10),
                tool_error("down").for_tool("weather"),
            ],
        ) as ctx:
            result = my_agent.run("...")
    """
    from agent_chaos.patch.patcher import ChaosPatcher

    injector = ChaosInjector(chaos=chaos)
    metrics = MetricsStore()

    session_id = ""
    if emit_events:
        from agent_chaos.ui.events import event_bus

        metrics.set_event_bus(event_bus)
        session_id = event_bus.start_session(name, description)
        metrics.set_trace_context(event_bus.trace_id, name)

    if event_sink is not None:
        metrics.set_event_sink(event_sink)
        if hasattr(event_sink, "start_trace") and callable(
            getattr(event_sink, "start_trace")
        ):
            trace_ctx = event_sink.start_trace(name)
            metrics.set_trace_context(trace_ctx.trace_id, trace_ctx.trace_name)
            session_id = trace_ctx.trace_id

    patcher = ChaosPatcher(injector, metrics)
    providers = providers or ["anthropic"]

    ctx = ChaosContext(
        name=name, injector=injector, metrics=metrics, session_id=session_id
    )
    injector.set_context(ctx)

    try:
        patcher.patch_providers(providers)
        yield ctx
    finally:
        patcher.unpatch_all()
        if emit_events:
            from agent_chaos.ui.events import event_bus

            event_bus.end_session()
        if event_sink is not None and hasattr(event_sink, "end_trace"):
            try:
                event_sink.end_trace(
                    metrics._trace_id,
                    metrics._trace_name,
                    {
                        "total_calls": metrics.total_calls,
                        "failed_calls": sum(
                            1
                            for c in metrics.call_history
                            if not c.get("success", True)
                        ),
                        "chaos_count": len(metrics.faults_injected),
                    },
                )
            except Exception:
                pass
        if event_sink is not None and hasattr(event_sink, "close"):
            try:
                event_sink.close()
            except Exception:
                pass
