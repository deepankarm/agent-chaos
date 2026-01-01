"""Tests for chaos/user.py - User input chaos types."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.user import UserInputMutateChaos, user_input_mutate
from agent_chaos.core.context import ChaosContext
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.types import ChaosAction


class TestUserInputMutateChaos:
    """Tests for UserInputMutateChaos."""

    def test_defaults(self) -> None:
        def my_fn(query: str) -> str:
            return query

        chaos = UserInputMutateChaos(mutator=my_fn)
        assert chaos.mutator == my_fn
        assert chaos.probability == 1.0
        assert chaos.always is True

    def test_point(self) -> None:
        def my_fn(query: str) -> str:
            return query

        chaos = UserInputMutateChaos(mutator=my_fn)
        assert chaos.point == ChaosPoint.USER_INPUT

    def test_apply_with_no_mutator_proceeds(self) -> None:
        chaos = UserInputMutateChaos(mutator=None)
        result = chaos.apply(query="hello")
        assert result.action == ChaosAction.PROCEED

    def test_apply_with_simple_mutator(self) -> None:
        def add_suffix(query: str) -> str:
            return query + " ATTACK"

        chaos = UserInputMutateChaos(mutator=add_suffix)
        result = chaos.apply(query="what is the weather")
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == "what is the weather ATTACK"

    def test_apply_with_ctx_mutator(self) -> None:
        def ctx_aware_mutate(ctx: ChaosContext, query: str) -> str:
            return f"[{ctx.name}] {query}"

        chaos = UserInputMutateChaos(mutator=ctx_aware_mutate)
        ctx = ChaosContext(
            name="test-ctx",
            injector=ChaosInjector(chaos=[]),
            metrics=MetricsStore(),
            session_id="test-123",
        )
        result = chaos.apply(ctx=ctx, query="hello world")
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == "[test-ctx] hello world"

    def test_should_trigger_on_turn(self) -> None:
        def my_fn(query: str) -> str:
            return query

        chaos = UserInputMutateChaos(mutator=my_fn, on_turn=2, always=False)
        assert not chaos.should_trigger(1, current_turn=1)
        assert chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(1, current_turn=3)

    def test_str_representation(self) -> None:
        def inject_attack(query: str) -> str:
            return query

        chaos = UserInputMutateChaos(mutator=inject_attack)
        assert str(chaos) == "user_input_mutate[inject_attack]"

    def test_str_with_no_mutator(self) -> None:
        chaos = UserInputMutateChaos(mutator=None)
        assert str(chaos) == "user_input_mutate[none]"


class TestUserInputMutateFactory:
    """Tests for user_input_mutate factory function."""

    def test_factory_creates_builder(self) -> None:
        def my_fn(query: str) -> str:
            return query.upper()

        builder = user_input_mutate(my_fn)
        chaos = builder.on_turn(3).build()
        assert isinstance(chaos, UserInputMutateChaos)
        assert chaos.on_turn == 3

    def test_factory_with_probability(self) -> None:
        def my_fn(query: str) -> str:
            return query

        builder = user_input_mutate(my_fn)
        chaos = builder.with_probability(0.5).build()
        assert chaos.probability == 0.5


class TestUserInputChaosTurnTriggers:
    """Tests for turn-based triggers on user input chaos."""

    def test_between_turns(self) -> None:
        # between_turns triggers when completed_turns >= after_turn AND current_turn == 0
        def my_fn(query: str) -> str:
            return query

        chaos = UserInputMutateChaos(mutator=my_fn, between_turns=(1, 3), always=False)
        # current_turn must be 0 (between turns, not during a turn)
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
