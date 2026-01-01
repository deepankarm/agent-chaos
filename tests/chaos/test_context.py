"""Tests for chaos/context.py - ContextMutateChaos."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.context import ContextMutateChaos, context_mutate
from agent_chaos.core.context import ChaosContext
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder
from agent_chaos.types import ChaosAction


class TestContextMutateChaos:
    """Tests for ContextMutateChaos."""

    def test_defaults(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn)
        assert chaos.mutator == my_fn
        assert chaos.on_call is None
        assert chaos.always is False

    def test_point(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn)
        assert chaos.point == ChaosPoint.MESSAGES

    def test_apply_with_no_mutator_proceeds(self) -> None:
        chaos = ContextMutateChaos(mutator=None)
        result = chaos.apply()
        assert result.action == ChaosAction.PROCEED

    def test_apply_with_simple_mutator(self) -> None:
        def add_message(messages: list) -> list:
            return messages + [{"role": "user", "content": "injected"}]

        chaos = ContextMutateChaos(mutator=add_message)
        messages = [{"role": "user", "content": "hello"}]
        result = chaos.apply(messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert len(result.mutated) == 2
        assert result.mutated[1]["content"] == "injected"

    def test_apply_with_ctx_mutator(self) -> None:
        def ctx_aware_mutate(ctx: ChaosContext, messages: list) -> list:
            return messages + [{"role": "system", "content": f"ctx: {ctx.name}"}]

        chaos = ContextMutateChaos(mutator=ctx_aware_mutate)
        ctx = ChaosContext(
            name="my-context",
            injector=ChaosInjector(chaos=[]),
            recorder=Recorder(metrics=MetricsStore()),
            session_id="test-123",
        )
        messages = [{"role": "user", "content": "hello"}]
        result = chaos.apply(ctx=ctx, messages=messages)
        assert result.action == ChaosAction.MUTATE
        assert result.mutated[-1]["content"] == "ctx: my-context"

    def test_should_trigger_on_call(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn, on_call=2)
        assert not chaos.should_trigger(1)
        assert chaos.should_trigger(2)
        assert not chaos.should_trigger(3)

    def test_should_trigger_always(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn, always=True)
        assert chaos.should_trigger(1)
        assert chaos.should_trigger(100)

    def test_str_representation(self) -> None:
        def my_mutator(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_mutator, on_call=3)
        assert str(chaos) == "context_mutate[my_mutator] on call 3"

    def test_str_representation_after_calls(self) -> None:
        def transform(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=transform, after_calls=5)
        assert str(chaos) == "context_mutate[transform] after 5 calls"


class TestContextMutateFactory:
    """Tests for context_mutate factory function."""

    def test_factory_creates_builder(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        builder = context_mutate(my_fn)
        chaos = builder.on_call(2).build()
        assert isinstance(chaos, ContextMutateChaos)
        assert chaos.on_call == 2

    def test_factory_with_always(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        builder = context_mutate(my_fn)
        chaos = builder.always().build()
        assert chaos.always is True


class TestContextChaosTurnTriggers:
    """Tests for turn-based triggers on context chaos."""

    def test_on_turn(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn, on_turn=2)
        assert not chaos.should_trigger(1, current_turn=1)
        assert chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(1, current_turn=3)

    def test_after_turns(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn, after_turns=2)
        assert not chaos.should_trigger(1, completed_turns=1)
        assert chaos.should_trigger(1, completed_turns=2)
        assert chaos.should_trigger(1, completed_turns=3)

    def test_between_turns(self) -> None:
        def my_fn(msgs: list) -> list:
            return msgs

        chaos = ContextMutateChaos(mutator=my_fn, between_turns=(2, 4))
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=2)
        assert not chaos.should_trigger(1, current_turn=0, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=3)
