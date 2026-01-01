"""Tests for Stream chaos types."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.stream import (
    SlowChunksChaos,
    SlowTTFTChaos,
    StreamCutChaos,
    StreamHangChaos,
    llm_slow_chunks,
    llm_slow_ttft,
    llm_stream_cut,
    llm_stream_hang,
)
from agent_chaos.types import ChaosAction


class TestStreamCutChaos:
    """Tests for StreamCutChaos."""

    def test_defaults(self) -> None:
        chaos = StreamCutChaos()
        assert chaos.after_chunks == 0
        assert chaos.probability == 1.0
        assert chaos.on_call is None
        assert chaos.always is False

    def test_custom_after_chunks(self) -> None:
        chaos = StreamCutChaos(after_chunks=10)
        assert chaos.after_chunks == 10

    def test_point(self) -> None:
        chaos = StreamCutChaos()
        assert chaos.point == ChaosPoint.STREAM

    def test_should_trigger_on_call(self) -> None:
        chaos = StreamCutChaos(on_call=2)
        assert not chaos.should_trigger(1)
        assert chaos.should_trigger(2)
        assert not chaos.should_trigger(3)

    def test_should_trigger_always(self) -> None:
        chaos = StreamCutChaos(always=True)
        assert chaos.should_trigger(1)
        assert chaos.should_trigger(100)

    def test_should_trigger_on_chunk(self) -> None:
        chaos = StreamCutChaos(after_chunks=5)
        assert not chaos.should_trigger_on_chunk(0)
        assert not chaos.should_trigger_on_chunk(4)
        assert chaos.should_trigger_on_chunk(5)
        assert chaos.should_trigger_on_chunk(10)

    def test_apply_returns_raise_action(self) -> None:
        chaos = StreamCutChaos()
        result = chaos.apply()
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_apply_exception_type(self) -> None:
        import anthropic

        chaos = StreamCutChaos()
        result = chaos.apply()
        assert isinstance(result.exception, anthropic.APIConnectionError)

    def test_str_representation(self) -> None:
        chaos = StreamCutChaos(after_chunks=15)
        assert str(chaos) == "stream_cut(after 15 chunks)"

    def test_provider_filter(self) -> None:
        chaos = StreamCutChaos(provider="anthropic", always=True)
        assert chaos.should_trigger(1, provider="anthropic")
        assert not chaos.should_trigger(1, provider="openai")


class TestStreamHangChaos:
    """Tests for StreamHangChaos."""

    def test_defaults(self) -> None:
        chaos = StreamHangChaos()
        assert chaos.after_chunks == 0
        assert chaos.hang_seconds == 60.0

    def test_custom_hang_seconds(self) -> None:
        chaos = StreamHangChaos(hang_seconds=120.0)
        assert chaos.hang_seconds == 120.0

    def test_point(self) -> None:
        chaos = StreamHangChaos()
        assert chaos.point == ChaosPoint.STREAM

    def test_should_trigger_on_chunk(self) -> None:
        chaos = StreamHangChaos(after_chunks=10)
        assert not chaos.should_trigger_on_chunk(5)
        assert chaos.should_trigger_on_chunk(10)
        assert chaos.should_trigger_on_chunk(15)

    def test_apply_returns_hang_action(self) -> None:
        chaos = StreamHangChaos()
        result = chaos.apply()
        assert result.action == ChaosAction.HANG

    def test_str_representation(self) -> None:
        chaos = StreamHangChaos(after_chunks=8)
        assert str(chaos) == "stream_hang(after 8 chunks)"


class TestSlowTTFTChaos:
    """Tests for SlowTTFTChaos."""

    def test_defaults(self) -> None:
        chaos = SlowTTFTChaos()
        assert chaos.delay == 0.0
        assert chaos.after_chunks == 0  # Always applies to first chunk

    def test_custom_delay(self) -> None:
        chaos = SlowTTFTChaos(delay=2.5)
        assert chaos.delay == 2.5

    def test_point(self) -> None:
        chaos = SlowTTFTChaos()
        assert chaos.point == ChaosPoint.STREAM

    def test_apply_returns_delay_action(self) -> None:
        chaos = SlowTTFTChaos(delay=1.5)
        result = chaos.apply()
        assert result.action == ChaosAction.DELAY
        assert result.mutated == 1.5

    def test_str_representation(self) -> None:
        chaos = SlowTTFTChaos(delay=3.0)
        assert str(chaos) == "slow_ttft(3.0s)"


class TestSlowChunksChaos:
    """Tests for SlowChunksChaos."""

    def test_defaults(self) -> None:
        chaos = SlowChunksChaos()
        assert chaos.delay == 0.0

    def test_custom_delay(self) -> None:
        chaos = SlowChunksChaos(delay=0.5)
        assert chaos.delay == 0.5

    def test_point(self) -> None:
        chaos = SlowChunksChaos()
        assert chaos.point == ChaosPoint.STREAM

    def test_apply_returns_delay_action(self) -> None:
        chaos = SlowChunksChaos(delay=0.3)
        result = chaos.apply()
        assert result.action == ChaosAction.DELAY
        assert result.mutated == 0.3

    def test_str_representation(self) -> None:
        chaos = SlowChunksChaos(delay=0.25)
        assert str(chaos) == "slow_chunks(0.25s)"


class TestStreamChaosFactories:
    """Tests for stream chaos factory functions."""

    def test_llm_stream_cut_factory(self) -> None:
        builder = llm_stream_cut(10)
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, StreamCutChaos)
        assert chaos.after_chunks == 10
        assert chaos.on_call == 1

    def test_llm_stream_hang_factory(self) -> None:
        builder = llm_stream_hang(5)
        chaos = builder.always().build()
        assert isinstance(chaos, StreamHangChaos)
        assert chaos.after_chunks == 5
        assert chaos.always is True

    def test_llm_slow_ttft_factory(self) -> None:
        builder = llm_slow_ttft(2.0)
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, SlowTTFTChaos)
        assert chaos.delay == 2.0

    def test_llm_slow_chunks_factory(self) -> None:
        builder = llm_slow_chunks(0.1)
        chaos = builder.after_calls(2).build()
        assert isinstance(chaos, SlowChunksChaos)
        assert chaos.delay == 0.1
        assert chaos.after_calls == 2


class TestStreamChaosTurnTriggers:
    """Tests for turn-based triggers on stream chaos."""

    def test_on_turn(self) -> None:
        chaos = StreamCutChaos(on_turn=3)
        assert not chaos.should_trigger(1, current_turn=1)
        assert not chaos.should_trigger(1, current_turn=2)
        assert chaos.should_trigger(1, current_turn=3)

    def test_after_turns(self) -> None:
        chaos = StreamHangChaos(after_turns=2)
        assert not chaos.should_trigger(1, completed_turns=1)
        # Triggers when completed_turns >= after_turns
        assert chaos.should_trigger(1, completed_turns=2)
        assert chaos.should_trigger(1, completed_turns=3)

    def test_between_turns(self) -> None:
        # between_turns triggers when completed_turns >= after_turn AND current_turn == 0
        chaos = SlowTTFTChaos(between_turns=(1, 3))
        # Between turns, current_turn must be 0
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
        assert not chaos.should_trigger(1, current_turn=0, completed_turns=0)


class TestStreamChaosProbability:
    """Tests for probability-based triggering on stream chaos."""

    def test_probability_zero_never_triggers(self) -> None:
        chaos = StreamCutChaos(after_chunks=5, probability=0.0)
        # With 0 probability, should never trigger
        triggered = False
        for _ in range(100):
            if chaos.should_trigger_on_chunk(10):
                triggered = True
                break
        assert not triggered

    def test_probability_one_always_triggers(self) -> None:
        chaos = StreamCutChaos(after_chunks=5, probability=1.0)
        # With 1.0 probability and enough chunks, should always trigger
        assert chaos.should_trigger_on_chunk(5)
        assert chaos.should_trigger_on_chunk(10)

    def test_probability_respects_after_chunks(self) -> None:
        chaos = StreamCutChaos(after_chunks=5, probability=1.0)
        # Before after_chunks, should not trigger
        assert not chaos.should_trigger_on_chunk(0)
        assert not chaos.should_trigger_on_chunk(4)
