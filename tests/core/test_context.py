"""Tests for core/context.py - ChaosContext class."""

from __future__ import annotations

import time
import pytest

from agent_chaos.core.context import ChaosContext
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder
from agent_chaos.scenario.model import TurnResult


@pytest.fixture
def ctx() -> ChaosContext:
    """Fresh ChaosContext for testing."""
    return ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        recorder=Recorder(metrics=MetricsStore()),
        session_id="test-123",
    )


class TestChaosContextInit:
    """Tests for ChaosContext initialization."""

    def test_basic_creation(self, ctx: ChaosContext) -> None:
        assert ctx.name == "test-ctx"
        assert ctx.session_id == "test-123"
        assert ctx.result is None
        assert ctx.error is None
        assert ctx.elapsed_s is None
        assert ctx.agent_input is None
        assert ctx.agent_output is None
        assert ctx.current_turn == 0
        assert ctx.turn_results == []
        assert ctx.agent_state == {}

    def test_with_injector(self) -> None:
        injector = ChaosInjector(chaos=[])
        ctx = ChaosContext(
            name="test",
            injector=injector,
            recorder=Recorder(metrics=MetricsStore()),
            session_id="session-1",
        )
        assert ctx.injector is injector

    def test_with_metrics(self) -> None:
        metrics = MetricsStore()
        recorder = Recorder(metrics=metrics)
        ctx = ChaosContext(
            name="test",
            injector=ChaosInjector(chaos=[]),
            recorder=recorder,
            session_id="session-1",
        )
        assert ctx.metrics is metrics


class TestChaosContextTurnTracking:
    """Tests for turn tracking methods."""

    def test_start_turn(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        assert ctx.current_turn == 1
        assert ctx._turn_start_calls == 0
        assert ctx.injector.current_turn == 1

    def test_start_turn_updates_metrics_turn(self, ctx: ChaosContext) -> None:
        ctx.start_turn(2, "Test input")
        # The metrics should have the current turn set
        assert ctx.metrics.conv.current_turn == 2

    def test_start_turn_resets_user_message_flag(self, ctx: ChaosContext) -> None:
        ctx.metrics.conv.user_message_recorded = True
        ctx.start_turn(1, "Hello")
        assert ctx.metrics.conv.user_message_recorded is False

    def test_end_turn_returns_turn_result(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        time.sleep(0.01)  # Small delay to ensure non-zero duration
        result = ctx.end_turn(
            turn_input="Hello",
            response="Hi there!",
            success=True,
        )
        assert isinstance(result, TurnResult)
        assert result.turn_number == 1
        assert result.input == "Hello"
        assert result.response == "Hi there!"
        assert result.success is True
        assert result.duration_s > 0

    def test_end_turn_appends_to_turn_results(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.end_turn("Hello", "Hi!", True)
        assert len(ctx.turn_results) == 1
        assert ctx.turn_results[0].turn_number == 1

    def test_end_turn_with_error(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Bad request")
        result = ctx.end_turn(
            turn_input="Bad request",
            response="",
            success=False,
            error="Rate limit exceeded",
        )
        assert result.success is False
        assert result.error == "Rate limit exceeded"

    def test_multiple_turns(self, ctx: ChaosContext) -> None:
        # Turn 1
        ctx.start_turn(1, "First")
        ctx.end_turn("First", "Response 1", True)

        # Turn 2
        ctx.start_turn(2, "Second")
        ctx.end_turn("Second", "Response 2", True)

        assert len(ctx.turn_results) == 2
        assert ctx.turn_results[0].turn_number == 1
        assert ctx.turn_results[1].turn_number == 2

    def test_end_turn_tracks_llm_calls(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        # Simulate some LLM calls
        ctx.metrics.calls.count = 3
        result = ctx.end_turn("Hello", "Hi!", True)
        assert result.llm_calls == 3

    def test_end_turn_tracks_tokens(self, ctx: ChaosContext) -> None:
        from agent_chaos.core.metrics import CallRecord

        ctx.start_turn(1, "Hello")
        # Simulate token usage via history
        ctx.metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 80, "output_tokens": 40},
            ),
        ]
        result = ctx.end_turn("Hello", "Hi!", True)
        assert result.input_tokens == 180
        assert result.output_tokens == 90
        assert result.total_tokens == 270


class TestChaosContextGetTurnResult:
    """Tests for get_turn_result method."""

    def test_get_existing_turn(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.end_turn("Hello", "Hi!", True)
        ctx.start_turn(2, "Next")
        ctx.end_turn("Next", "Response", True)

        result = ctx.get_turn_result(1)
        assert result is not None
        assert result.turn_number == 1

        result = ctx.get_turn_result(2)
        assert result is not None
        assert result.turn_number == 2

    def test_get_nonexistent_turn(self, ctx: ChaosContext) -> None:
        result = ctx.get_turn_result(99)
        assert result is None


class TestChaosContextMessageHistory:
    """Tests for get_message_history method."""

    def test_empty_history(self, ctx: ChaosContext) -> None:
        history = ctx.get_message_history()
        assert history == []

    def test_history_with_successful_turns(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.end_turn("Hello", "Hi there!", True)
        ctx.start_turn(2, "How are you?")
        ctx.end_turn("How are you?", "I'm great!", True)

        history = ctx.get_message_history()
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}
        assert history[2] == {"role": "user", "content": "How are you?"}
        assert history[3] == {"role": "assistant", "content": "I'm great!"}

    def test_history_includes_user_messages_from_failed_turns(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.end_turn("Hello", "Hi!", True)
        ctx.start_turn(2, "Do something")
        ctx.end_turn("Do something", "", False, error="Rate limit")
        ctx.start_turn(3, "Retry")
        ctx.end_turn("Retry", "Done!", True)

        history = ctx.get_message_history()
        # Should have: user1, asst1, user2 (failed, no asst), user3, asst3
        assert len(history) == 5
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"  # Failed turn user message
        assert history[3]["role"] == "user"  # Next turn
        assert history[4]["role"] == "assistant"

    def test_history_excludes_assistant_from_failed_turns(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.end_turn("Hello", "", False, error="Failed")

        history = ctx.get_message_history()
        # Only the user message, no assistant
        assert len(history) == 1
        assert history[0]["role"] == "user"


class TestChaosContextAgentState:
    """Tests for agent_state dictionary."""

    def test_agent_state_empty_by_default(self, ctx: ChaosContext) -> None:
        assert ctx.agent_state == {}

    def test_agent_state_can_store_data(self, ctx: ChaosContext) -> None:
        ctx.agent_state["message_history"] = ["msg1", "msg2"]
        ctx.agent_state["memory"] = {"key": "value"}
        assert ctx.agent_state["message_history"] == ["msg1", "msg2"]
        assert ctx.agent_state["memory"] == {"key": "value"}

    def test_agent_state_persists_across_turns(self, ctx: ChaosContext) -> None:
        ctx.start_turn(1, "Hello")
        ctx.agent_state["counter"] = 1
        ctx.end_turn("Hello", "Hi!", True)

        ctx.start_turn(2, "Next")
        ctx.agent_state["counter"] = ctx.agent_state.get("counter", 0) + 1
        ctx.end_turn("Next", "Done!", True)

        assert ctx.agent_state["counter"] == 2


class TestChaosContextWithFixtures:
    """Tests using conftest fixtures."""

    def test_basic_init(self, chaos_injector: ChaosInjector, recorder: Recorder, metrics_store: MetricsStore):
        """Test basic context creation."""
        ctx = ChaosContext(
            name="test",
            injector=chaos_injector,
            recorder=recorder,
            session_id="session-123",
        )
        assert ctx.name == "test"
        assert ctx.session_id == "session-123"
        assert ctx.injector is chaos_injector
        assert ctx.metrics is metrics_store

    def test_initial_state(self, chaos_context: ChaosContext):
        """Test initial state values."""
        assert chaos_context.result is None
        assert chaos_context.error is None
        assert chaos_context.elapsed_s is None
        assert chaos_context.current_turn == 0
        assert chaos_context.turn_results == []
        assert chaos_context.agent_state == {}
