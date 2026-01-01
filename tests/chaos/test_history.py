"""Tests for chaos/history.py - History chaos types."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.history import (
    HistoryInjectChaos,
    HistoryMutateChaos,
    HistoryTruncateChaos,
    history_inject,
    history_mutate,
    history_truncate,
)
from agent_chaos.core.context import ChaosContext
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.types import ChaosAction


class TestHistoryMutateChaos:
    """Tests for HistoryMutateChaos."""

    def test_defaults(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = HistoryMutateChaos(mutator=my_fn)
        assert chaos.on_call is None
        assert chaos.always is False

    def test_point(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = HistoryMutateChaos(mutator=my_fn)
        assert chaos.point == ChaosPoint.MESSAGES

    def test_apply_with_none_messages(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs + [{"role": "user", "content": "extra"}]

        chaos = HistoryMutateChaos(mutator=my_fn)
        result = chaos.apply(messages=None)
        assert result.action == ChaosAction.PROCEED

    def test_apply_with_simple_mutator(self) -> None:
        def add_fake(msgs: list) -> list:
            return msgs + [{"role": "user", "content": "fake message"}]

        chaos = HistoryMutateChaos(mutator=add_fake)
        messages = [{"role": "user", "content": "original"}]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert len(result.mutated) == 2
        assert result.mutated[1]["content"] == "fake message"

    def test_apply_with_ctx_mutator(self) -> None:
        def ctx_mutator(ctx: ChaosContext, msgs: list) -> list:
            return msgs + [{"role": "system", "content": ctx.session_id}]

        chaos = HistoryMutateChaos(mutator=ctx_mutator)
        ctx = ChaosContext(
            name="test",
            injector=ChaosInjector(chaos=[]),
            metrics=MetricsStore(),
            session_id="session-abc",
        )
        result = chaos.apply(ctx=ctx, messages=[])
        assert result.mutated[-1]["content"] == "session-abc"

    def test_should_trigger_between_turns(self) -> None:
        # between_turns triggers when completed_turns >= after_turn AND current_turn == 0
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = HistoryMutateChaos(mutator=my_fn, between_turns=(1, 3))
        # Needs current_turn=0 (between turns, not during a turn)
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)


class TestHistoryTruncateChaos:
    """Tests for HistoryTruncateChaos."""

    def test_defaults(self) -> None:
        chaos = HistoryTruncateChaos()
        assert chaos.keep_last == 3
        assert chaos.keep_system is True

    def test_point(self) -> None:
        chaos = HistoryTruncateChaos()
        assert chaos.point == ChaosPoint.MESSAGES

    def test_apply_with_none_messages(self) -> None:
        chaos = HistoryTruncateChaos()
        result = chaos.apply(messages=None)
        assert result.action == ChaosAction.PROCEED

    def test_apply_with_fewer_messages(self) -> None:
        chaos = HistoryTruncateChaos(keep_last=5)
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.PROCEED

    def test_apply_truncates_to_keep_last(self) -> None:
        chaos = HistoryTruncateChaos(keep_last=2, keep_system=False)
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
        ]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert len(result.mutated) == 2
        assert result.mutated[0]["content"] == "msg4"
        assert result.mutated[1]["content"] == "msg5"

    def test_apply_keeps_system_messages(self) -> None:
        chaos = HistoryTruncateChaos(keep_last=2, keep_system=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        # System + last 2 non-system
        assert len(result.mutated) == 3
        assert result.mutated[0]["role"] == "system"
        assert result.mutated[1]["content"] == "msg3"
        assert result.mutated[2]["content"] == "msg4"


class TestHistoryInjectChaos:
    """Tests for HistoryInjectChaos."""

    def test_defaults(self) -> None:
        injected = [{"role": "user", "content": "fake"}]
        chaos = HistoryInjectChaos(messages=injected)
        assert chaos.position == "end"

    def test_point(self) -> None:
        chaos = HistoryInjectChaos(messages=[])
        assert chaos.point == ChaosPoint.MESSAGES

    def test_apply_with_none_messages(self) -> None:
        injected = [{"role": "user", "content": "fake"}]
        chaos = HistoryInjectChaos(messages=injected)
        result = chaos.apply(messages=None)
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == injected

    def test_apply_inject_at_end(self) -> None:
        injected = [{"role": "user", "content": "injected"}]
        chaos = HistoryInjectChaos(messages=injected, position="end")
        messages = [{"role": "user", "content": "original"}]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert len(result.mutated) == 2
        assert result.mutated[0]["content"] == "original"
        assert result.mutated[1]["content"] == "injected"

    def test_apply_inject_at_start(self) -> None:
        injected = [{"role": "user", "content": "injected"}]
        chaos = HistoryInjectChaos(messages=injected, position="start")
        messages = [{"role": "user", "content": "original"}]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert len(result.mutated) == 2
        assert result.mutated[0]["content"] == "injected"
        assert result.mutated[1]["content"] == "original"

    def test_apply_inject_at_random(self) -> None:
        injected = [{"role": "user", "content": "injected"}]
        chaos = HistoryInjectChaos(messages=injected, position="random")
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        # Should have 3 messages total
        assert len(result.mutated) == 3
        # The injected message should be somewhere
        contents = [m["content"] for m in result.mutated]
        assert "injected" in contents


class TestHistoryChaosFactories:
    """Tests for history chaos factory functions."""

    def test_history_mutate_factory(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs[::-1]

        builder = history_mutate(my_fn)
        chaos = builder.between_turns(1, 2).build()
        assert isinstance(chaos, HistoryMutateChaos)
        assert chaos.between_turns == (1, 2)

    def test_history_truncate_factory(self) -> None:
        builder = history_truncate(keep_last=5, keep_system=False)
        chaos = builder.between_turns(2, 3).build()
        assert isinstance(chaos, HistoryTruncateChaos)
        assert chaos.keep_last == 5
        assert chaos.keep_system is False

    def test_history_inject_factory(self) -> None:
        msgs = [{"role": "user", "content": "I already paid!"}]
        builder = history_inject(msgs, position="start")
        chaos = builder.between_turns(1, 2).build()
        assert isinstance(chaos, HistoryInjectChaos)
        assert chaos.messages == msgs
        assert chaos.position == "start"
