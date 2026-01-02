"""Chaos context manager and context object."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder

if TYPE_CHECKING:
    from agent_chaos.scenario.model import TurnResult


class ChaosContext:
    """Context object providing access to injector, recorder and metrics."""

    def __init__(
        self,
        name: str,
        injector: ChaosInjector,
        recorder: Recorder,
        session_id: str,
    ):
        self.name = name
        self.injector = injector
        self.recorder = recorder
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
        self._turn_start_call_history_len: int = 0  # call_history length at turn start

        # Agent state - persists across turns for framework-specific data
        # (e.g., pydantic-ai message_history, langchain memory, etc.)
        self.agent_state: dict[str, Any] = {}

    @property
    def metrics(self) -> MetricsStore:
        """Access to MetricsStore for data operations.

        For event emission, use self.recorder methods instead.
        """
        return self.recorder.metrics  # type: ignore[return-value]

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
        self._turn_start_call_history_len = len(self.metrics.history)

        # Reset user message flag so each turn can record its user message
        self.metrics.reset_user_message_flag()

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

        # Calculate tokens used during this turn
        input_tokens = 0
        output_tokens = 0
        for call in self.metrics.history[self._turn_start_call_history_len:]:
            input_tokens += call.usage.get("input_tokens") or 0
            output_tokens += call.usage.get("output_tokens") or 0
        total_tokens = input_tokens + output_tokens

        turn_result = TurnResult(
            turn_number=self.current_turn,
            input=turn_input,
            response=response,
            success=success,
            duration_s=duration_s,
            llm_calls=llm_calls,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
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

    def get_message_history(self) -> list[dict[str, str]]:
        """Get conversation history from previous turns.

        Returns a list of messages in a simple format that agents can use
        to build context for LLM calls. This includes ALL previous turns,
        even those that failed (so user context is preserved).

        Returns:
            List of dicts with 'role' ('user' or 'assistant') and 'content'.

        Example:
            >>> history = ctx.get_message_history()
            >>> # [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        messages = []
        for turn in self.turn_results:
            # Always include user message (even for failed turns)
            if turn.input:
                messages.append({"role": "user", "content": turn.input})
            # Only include assistant response if turn succeeded
            if turn.success and turn.response:
                messages.append({"role": "assistant", "content": turn.response})
        return messages


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
        ChaosContext with injector, recorder and metrics access

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
    from agent_chaos.events.sink import EventSink, MultiSink
    from agent_chaos.patch.patcher import ChaosPatcher

    injector = ChaosInjector(chaos=chaos)
    metrics = MetricsStore()

    # Build composite sink from enabled sinks
    sinks: list[EventSink] = []
    session_id = ""

    if emit_events:
        from agent_chaos.events.ui_sink import UISink
        from agent_chaos.ui.events import event_bus

        sinks.append(UISink(event_bus))
        session_id = event_bus.start_session(name, description)

    if event_sink is not None:
        sinks.append(event_sink)

    # Create recorder with composite sink
    sink = MultiSink(sinks) if sinks else None
    recorder = Recorder(sink=sink, metrics=metrics)

    # Start trace if we have any sinks
    if sinks:
        trace_id = recorder.start_trace(name, description)
        session_id = session_id or trace_id

    patcher = ChaosPatcher(injector, recorder)
    providers = providers or ["anthropic"]

    ctx = ChaosContext(
        name=name, injector=injector, recorder=recorder, session_id=session_id
    )
    injector.set_context(ctx)

    try:
        patcher.patch_providers(providers)
        yield ctx
    finally:
        patcher.unpatch_all()

        # End trace if we started one
        if sinks:
            recorder.end_trace(success=(ctx.error is None))

        if emit_events:
            from agent_chaos.ui.events import event_bus
            event_bus.end_session()

        # Close the recorder (which closes all sinks)
        recorder.close()
