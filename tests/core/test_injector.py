"""Tests for core/injector.py - ChaosInjector class."""

from __future__ import annotations

import pytest

from agent_chaos.core.injector import ChaosInjector
from agent_chaos.chaos.llm import RateLimitChaos, TimeoutChaos, ServerErrorChaos
from agent_chaos.chaos.stream import (
    SlowTTFTChaos,
    StreamCutChaos,
    StreamHangChaos,
    SlowChunksChaos,
)
from agent_chaos.chaos.tool import ToolErrorChaos, ToolEmptyChaos, ToolMutateChaos
from agent_chaos.chaos.context import ContextMutateChaos
from agent_chaos.chaos.user import UserInputMutateChaos
from agent_chaos.types import ChaosAction


class TestChaosInjectorInitialization:
    """Tests for ChaosInjector initialization."""

    def test_empty_initialization(self) -> None:
        injector = ChaosInjector()
        assert injector._call_count == 0
        assert injector._current_turn == 0
        assert injector._completed_turns == 0

    def test_initialization_with_chaos_list(self) -> None:
        chaos = [RateLimitChaos(on_call=1), TimeoutChaos(on_call=2)]
        injector = ChaosInjector(chaos=chaos)
        assert len(injector._llm_chaos) == 2

    def test_initialization_routes_chaos_by_point(self) -> None:
        chaos = [
            RateLimitChaos(on_call=1),  # LLM_CALL
            StreamCutChaos(after_chunks=5),  # STREAM
            ToolErrorChaos(always=True),  # TOOL_RESULT
        ]
        injector = ChaosInjector(chaos=chaos)
        assert len(injector._llm_chaos) == 1
        assert len(injector._stream_chaos) == 1
        assert len(injector._tool_chaos) == 1


class TestChaosInjectorCallTracking:
    """Tests for call tracking."""

    def test_increment_call(self) -> None:
        injector = ChaosInjector()
        assert injector._call_count == 0
        count = injector.increment_call()
        assert count == 1
        assert injector._call_count == 1

    def test_multiple_increments(self) -> None:
        injector = ChaosInjector()
        for i in range(5):
            count = injector.increment_call()
            assert count == i + 1


class TestChaosInjectorTurnTracking:
    """Tests for turn tracking."""

    def test_set_current_turn(self) -> None:
        injector = ChaosInjector()
        injector.set_current_turn(3)
        assert injector.current_turn == 3

    def test_complete_turn(self) -> None:
        injector = ChaosInjector()
        injector.set_current_turn(2)
        injector.complete_turn()
        assert injector.completed_turns == 2

    def test_completed_turns_property(self) -> None:
        injector = ChaosInjector()
        injector.set_current_turn(1)
        injector.complete_turn()
        injector.set_current_turn(2)
        injector.complete_turn()
        assert injector.completed_turns == 2


class TestChaosInjectorLLMChaos:
    """Tests for LLM chaos injection."""

    def test_next_llm_chaos_returns_none_when_empty(self) -> None:
        injector = ChaosInjector()
        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is None

    def test_next_llm_chaos_triggers_on_call(self) -> None:
        chaos = RateLimitChaos(on_call=2)
        injector = ChaosInjector(chaos=[chaos])

        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is None

        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is not None
        assert result.action == ChaosAction.RAISE

    def test_chaos_used_once(self) -> None:
        chaos = RateLimitChaos(on_call=1)
        injector = ChaosInjector(chaos=[chaos])

        injector.increment_call()
        result1 = injector.next_llm_chaos("anthropic")
        assert result1 is not None

        # Same call number, chaos already used
        result2 = injector.next_llm_chaos("anthropic")
        assert result2 is None

    def test_multiple_chaos_in_order(self) -> None:
        chaos1 = RateLimitChaos(on_call=1)
        chaos2 = TimeoutChaos(on_call=2)
        injector = ChaosInjector(chaos=[chaos1, chaos2])

        injector.increment_call()
        result1 = injector.next_llm_chaos("anthropic")
        assert result1 is not None  # RateLimitChaos

        injector.increment_call()
        result2 = injector.next_llm_chaos("anthropic")
        assert result2 is not None  # TimeoutChaos

    def test_injector_after_calls(self) -> None:
        """Test chaos triggering after N calls."""
        chaos = TimeoutChaos(after_calls=2)
        injector = ChaosInjector(chaos=[chaos])

        # Calls 1-2: no chaos
        injector.increment_call()
        assert injector.next_llm_chaos("anthropic") is None
        injector.increment_call()
        assert injector.next_llm_chaos("anthropic") is None

        # Call 3: chaos triggers (after 2 calls)
        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is not None, "Call 3 should trigger chaos"


class TestChaosInjectorStreamChaos:
    """Tests for stream chaos methods."""

    def test_get_stream_chaos(self) -> None:
        chaos = [
            StreamCutChaos(after_chunks=5),
            StreamHangChaos(after_chunks=10),
        ]
        injector = ChaosInjector(chaos=chaos)
        stream_chaos = injector.get_stream_chaos()
        assert len(stream_chaos) == 2

    def test_ttft_delay_with_slow_ttft(self) -> None:
        chaos = SlowTTFTChaos(delay=2.5)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.ttft_delay() == 2.5

    def test_ttft_delay_without_slow_ttft(self) -> None:
        chaos = StreamCutChaos(after_chunks=5)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.ttft_delay() is None

    def test_should_hang(self) -> None:
        chaos = StreamHangChaos(after_chunks=10)
        injector = ChaosInjector(chaos=[chaos])
        assert not injector.should_hang(5)
        assert injector.should_hang(10)
        assert injector.should_hang(15)

    def test_should_cut(self) -> None:
        chaos = StreamCutChaos(after_chunks=5)
        injector = ChaosInjector(chaos=[chaos])
        assert not injector.should_cut(3)
        assert injector.should_cut(5)
        assert injector.should_cut(10)

    def test_chunk_delay_with_slow_chunks(self) -> None:
        chaos = SlowChunksChaos(delay=0.5)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.chunk_delay() == 0.5

    def test_chunk_delay_without_slow_chunks(self) -> None:
        chaos = StreamCutChaos(after_chunks=5)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.chunk_delay() is None

    def test_stream_chaos_combined(self) -> None:
        """Test stream chaos detection methods."""
        injector = ChaosInjector(
            chaos=[
                SlowTTFTChaos(delay=1.5),
                StreamCutChaos(after_chunks=10),
                StreamHangChaos(after_chunks=20),
            ]
        )

        # TTFT delay
        assert injector.ttft_delay() == 1.5

        # Stream cut
        assert not injector.should_cut(5)
        assert injector.should_cut(10)
        assert injector.should_cut(15)

        # Stream hang
        assert not injector.should_hang(10)
        assert injector.should_hang(20)


class TestChaosInjectorToolChaos:
    """Tests for tool chaos methods."""

    def test_should_mutate_tools_true(self) -> None:
        chaos = ToolErrorChaos(always=True)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.should_mutate_tools() is True

    def test_should_mutate_tools_false(self) -> None:
        chaos = RateLimitChaos(on_call=1)  # LLM chaos, not tool
        injector = ChaosInjector(chaos=[chaos])
        assert injector.should_mutate_tools() is False

    def test_next_tool_chaos_triggers(self) -> None:
        chaos = ToolErrorChaos(always=True)
        injector = ChaosInjector(chaos=[chaos])

        result = injector.next_tool_chaos("weather", '{"temp": 72}')
        assert result is not None
        chaos_result, chaos_obj = result
        assert chaos_result.action == ChaosAction.MUTATE
        assert isinstance(chaos_obj, ToolErrorChaos)

    def test_next_tool_chaos_with_tool_filter(self) -> None:
        chaos = ToolErrorChaos(tool_name="weather", always=True)
        injector = ChaosInjector(chaos=[chaos])

        # Should trigger for weather
        result = injector.next_tool_chaos("weather", '{"temp": 72}')
        assert result is not None

        # Reset to test different tool - create new injector
        injector = ChaosInjector(chaos=[ToolErrorChaos(tool_name="weather", always=True)])

        # Should NOT trigger for calculator
        result = injector.next_tool_chaos("calculator", "42")
        assert result is None

    def test_tool_mutation_tracking(self) -> None:
        injector = ChaosInjector()

        assert not injector.is_tool_already_mutated("tool_123")

        injector.mark_tool_mutated("tool_123")
        assert injector.is_tool_already_mutated("tool_123")
        assert not injector.is_tool_already_mutated("tool_456")


class TestChaosInjectorContextChaos:
    """Tests for context chaos methods."""

    def test_next_context_chaos_triggers(self) -> None:
        def mutate_messages(msgs: list) -> list:
            return msgs + [{"role": "user", "content": "extra"}]

        chaos = ContextMutateChaos(mutator=mutate_messages, always=True)
        injector = ChaosInjector(chaos=[chaos])

        messages = [{"role": "user", "content": "hello"}]
        result = injector.next_context_chaos(messages)

        assert result is not None
        chaos_result, chaos_obj = result
        assert chaos_result.action == ChaosAction.MUTATE
        assert len(chaos_result.mutated) == 2

    def test_next_context_chaos_returns_none_when_empty(self) -> None:
        injector = ChaosInjector()
        result = injector.next_context_chaos([])
        assert result is None


class TestChaosInjectorUserChaos:
    """Tests for user input chaos methods."""

    def test_has_user_chaos_true(self) -> None:
        def mutate_query(q: str) -> str:
            return q.upper()

        chaos = UserInputMutateChaos(mutator=mutate_query)
        injector = ChaosInjector(chaos=[chaos])
        assert injector.has_user_chaos() is True

    def test_has_user_chaos_false(self) -> None:
        chaos = RateLimitChaos(on_call=1)  # Not user chaos
        injector = ChaosInjector(chaos=[chaos])
        assert injector.has_user_chaos() is False

    def test_apply_user_chaos_mutates_query(self) -> None:
        def mutate_query(q: str) -> str:
            return q.upper()

        chaos = UserInputMutateChaos(mutator=mutate_query)
        injector = ChaosInjector(chaos=[chaos])

        mutated, used_chaos = injector.apply_user_chaos("hello world")
        assert mutated == "HELLO WORLD"
        assert used_chaos is not None

    def test_apply_user_chaos_only_once(self) -> None:
        def mutate_query(q: str) -> str:
            return q.upper()

        chaos = UserInputMutateChaos(mutator=mutate_query)
        injector = ChaosInjector(chaos=[chaos])

        # First call applies chaos
        mutated1, used_chaos1 = injector.apply_user_chaos("hello")
        assert mutated1 == "HELLO"
        assert used_chaos1 is not None

        # Second call does not apply chaos again
        mutated2, used_chaos2 = injector.apply_user_chaos("world")
        assert mutated2 == "world"  # Not mutated
        assert used_chaos2 is None

    def test_apply_user_chaos_returns_original_when_no_chaos(self) -> None:
        injector = ChaosInjector()
        mutated, used_chaos = injector.apply_user_chaos("hello")
        assert mutated == "hello"
        assert used_chaos is None


class TestChaosInjectorProviderTargeting:
    """Tests for provider-specific chaos targeting."""

    def test_provider_targeting_anthropic(self) -> None:
        anthropic_chaos = RateLimitChaos(on_call=1, provider="anthropic")
        other_chaos = RateLimitChaos(on_call=1, provider="other_provider")
        injector = ChaosInjector(chaos=[anthropic_chaos, other_chaos])

        # Anthropic call should get anthropic chaos
        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is not None

    def test_provider_targeting_no_match(self) -> None:
        anthropic_chaos = RateLimitChaos(on_call=1, provider="anthropic")
        other_chaos = RateLimitChaos(on_call=1, provider="other_provider")
        injector = ChaosInjector(chaos=[anthropic_chaos, other_chaos])

        # Different provider call should NOT get provider-specific chaos
        injector.increment_call()
        result = injector.next_llm_chaos("some_other_provider")
        assert result is None


class TestChaosInjectorContext:
    """Tests for ChaosContext integration."""

    def test_set_context(self) -> None:
        from agent_chaos.core.context import ChaosContext
        from agent_chaos.core.metrics import MetricsStore
        from agent_chaos.core.recorder import Recorder

        injector = ChaosInjector()
        ctx = ChaosContext(
            name="test",
            injector=injector,
            recorder=Recorder(metrics=MetricsStore()),
            session_id="test-123",
        )

        injector.set_context(ctx)
        assert injector._ctx == ctx


class TestChaosInjectorBuilderSupport:
    """Tests for ChaosBuilder integration."""

    def test_accepts_builder_in_chaos_list(self) -> None:
        from agent_chaos.chaos.llm import llm_rate_limit

        # Pass a builder instead of a built chaos
        builder = llm_rate_limit().on_call(1)
        injector = ChaosInjector(chaos=[builder])

        # Should work just like passing built chaos
        injector.increment_call()
        result = injector.next_llm_chaos("anthropic")
        assert result is not None
