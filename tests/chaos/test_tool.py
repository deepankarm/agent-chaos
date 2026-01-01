"""Tests for Tool chaos types."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.tool import (
    ToolEmptyChaos,
    ToolErrorChaos,
    ToolMutateChaos,
    ToolTimeoutChaos,
    tool_empty,
    tool_error,
    tool_mutate,
    tool_timeout,
)
from agent_chaos.types import ChaosAction


class TestToolErrorChaos:
    """Tests for ToolErrorChaos."""

    def test_defaults(self) -> None:
        chaos = ToolErrorChaos()
        assert chaos.error_message == "Tool error"
        assert chaos.tool_name is None
        assert chaos.on_call is None
        assert chaos.always is False

    def test_custom_error_message(self) -> None:
        chaos = ToolErrorChaos(error_message="Custom error")
        assert chaos.error_message == "Custom error"

    def test_tool_name_filter(self) -> None:
        chaos = ToolErrorChaos(tool_name="weather", always=True)
        assert chaos.tool_name == "weather"

    def test_point(self) -> None:
        chaos = ToolErrorChaos()
        assert chaos.point == ChaosPoint.TOOL_RESULT

    def test_should_trigger_with_tool_filter(self) -> None:
        chaos = ToolErrorChaos(tool_name="weather", always=True)
        assert chaos.should_trigger(1, tool_name="weather")
        assert not chaos.should_trigger(1, tool_name="calculator")

    def test_should_trigger_without_tool_filter(self) -> None:
        chaos = ToolErrorChaos(always=True)
        assert chaos.should_trigger(1, tool_name="weather")
        assert chaos.should_trigger(1, tool_name="calculator")
        assert chaos.should_trigger(1, tool_name="anything")

    def test_should_trigger_on_call(self) -> None:
        chaos = ToolErrorChaos(on_call=2)
        assert not chaos.should_trigger(1)
        assert chaos.should_trigger(2)
        assert not chaos.should_trigger(3)

    def test_apply_returns_mutate_with_error(self) -> None:
        chaos = ToolErrorChaos(error_message="API failure")
        result = chaos.apply()
        assert result.action == ChaosAction.MUTATE
        assert '{"error": "API failure"}' == result.mutated

    def test_str_representation_all_tools(self) -> None:
        chaos = ToolErrorChaos()
        assert str(chaos) == "tool_error(all)"

    def test_str_representation_specific_tool(self) -> None:
        chaos = ToolErrorChaos(tool_name="weather")
        assert str(chaos) == "tool_error(weather)"


class TestToolEmptyChaos:
    """Tests for ToolEmptyChaos."""

    def test_defaults(self) -> None:
        chaos = ToolEmptyChaos()
        assert chaos.tool_name is None

    def test_point(self) -> None:
        chaos = ToolEmptyChaos()
        assert chaos.point == ChaosPoint.TOOL_RESULT

    def test_apply_returns_empty_string(self) -> None:
        chaos = ToolEmptyChaos()
        result = chaos.apply()
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == ""

    def test_str_representation_all_tools(self) -> None:
        chaos = ToolEmptyChaos()
        assert str(chaos) == "tool_empty(all)"

    def test_str_representation_specific_tool(self) -> None:
        chaos = ToolEmptyChaos(tool_name="search")
        assert str(chaos) == "tool_empty(search)"


class TestToolTimeoutChaos:
    """Tests for ToolTimeoutChaos."""

    def test_defaults(self) -> None:
        chaos = ToolTimeoutChaos()
        assert chaos.timeout_seconds == 30.0
        assert chaos.tool_name is None

    def test_custom_timeout(self) -> None:
        chaos = ToolTimeoutChaos(timeout_seconds=60.0)
        assert chaos.timeout_seconds == 60.0

    def test_point(self) -> None:
        chaos = ToolTimeoutChaos()
        assert chaos.point == ChaosPoint.TOOL_RESULT

    def test_apply_returns_timeout_message(self) -> None:
        chaos = ToolTimeoutChaos(timeout_seconds=45.0)
        result = chaos.apply()
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == "Tool execution timed out after 45.0s"

    def test_str_representation_all_tools(self) -> None:
        chaos = ToolTimeoutChaos(timeout_seconds=30.0)
        assert str(chaos) == "tool_timeout(30.0s)(all)"

    def test_str_representation_specific_tool(self) -> None:
        chaos = ToolTimeoutChaos(timeout_seconds=15.0, tool_name="database")
        assert str(chaos) == "tool_timeout(15.0s)(database)"


class TestToolMutateChaos:
    """Tests for ToolMutateChaos."""

    def test_defaults(self) -> None:
        chaos = ToolMutateChaos()
        assert chaos.mutator is None
        assert chaos.tool_name is None

    def test_point(self) -> None:
        chaos = ToolMutateChaos()
        assert chaos.point == ChaosPoint.TOOL_RESULT

    def test_apply_with_no_mutator_proceeds(self) -> None:
        chaos = ToolMutateChaos()
        result = chaos.apply()
        assert result.action == ChaosAction.PROCEED

    def test_apply_with_simple_mutator(self) -> None:
        def my_mutator(tool_name: str, result: str) -> str:
            return f"MUTATED({tool_name}): {result}"

        chaos = ToolMutateChaos(mutator=my_mutator)
        result = chaos.apply(tool_name="weather", result='{"temp": 72}')
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == 'MUTATED(weather): {"temp": 72}'

    def test_apply_with_ctx_mutator(self) -> None:
        from agent_chaos.core.context import ChaosContext
        from agent_chaos.core.injector import ChaosInjector
        from agent_chaos.core.metrics import MetricsStore

        def ctx_mutator(ctx: ChaosContext, tool_name: str, result: str) -> str:
            return f"[{ctx.name}] {tool_name}: {result}"

        chaos = ToolMutateChaos(mutator=ctx_mutator)
        ctx = ChaosContext(
            name="test-ctx",
            injector=ChaosInjector(chaos=[]),
            metrics=MetricsStore(),
            session_id="test-123",
        )
        result = chaos.apply(ctx=ctx, tool_name="calc", result="42")
        assert result.action == ChaosAction.MUTATE
        assert result.mutated == "[test-ctx] calc: 42"

    def test_str_representation_all_tools(self) -> None:
        def my_fn(tool_name: str, result: str) -> str:
            return result

        chaos = ToolMutateChaos(mutator=my_fn)
        assert str(chaos) == "tool_mutate[my_fn](all)"

    def test_str_representation_specific_tool(self) -> None:
        def transform(tool_name: str, result: str) -> str:
            return result.upper()

        chaos = ToolMutateChaos(mutator=transform, tool_name="weather")
        assert str(chaos) == "tool_mutate[transform](weather)"


class TestToolChaosFactories:
    """Tests for tool chaos factory functions."""

    def test_tool_error_factory(self) -> None:
        builder = tool_error("Connection failed")
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, ToolErrorChaos)
        assert chaos.error_message == "Connection failed"
        assert chaos.on_call == 1

    def test_tool_error_factory_default_message(self) -> None:
        builder = tool_error()
        chaos = builder.always().build()
        assert isinstance(chaos, ToolErrorChaos)
        assert chaos.error_message == "Tool error"

    def test_tool_empty_factory(self) -> None:
        builder = tool_empty()
        chaos = builder.after_calls(3).build()
        assert isinstance(chaos, ToolEmptyChaos)
        assert chaos.after_calls == 3

    def test_tool_timeout_factory(self) -> None:
        builder = tool_timeout(45.0)
        chaos = builder.on_call(2).build()
        assert isinstance(chaos, ToolTimeoutChaos)
        assert chaos.timeout_seconds == 45.0
        assert chaos.on_call == 2

    def test_tool_timeout_factory_default(self) -> None:
        builder = tool_timeout()
        chaos = builder.always().build()
        assert isinstance(chaos, ToolTimeoutChaos)
        assert chaos.timeout_seconds == 30.0

    def test_tool_mutate_factory(self) -> None:
        def double(tool_name: str, result: str) -> str:
            return result + result

        builder = tool_mutate(double)
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, ToolMutateChaos)
        assert chaos.mutator == double


class TestToolChaosTurnTriggers:
    """Tests for turn-based triggers on tool chaos."""

    def test_on_turn(self) -> None:
        chaos = ToolErrorChaos(on_turn=2)
        assert not chaos.should_trigger(1, current_turn=1)
        assert chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(1, current_turn=3)

    def test_after_turns(self) -> None:
        chaos = ToolEmptyChaos(after_turns=3)
        assert not chaos.should_trigger(1, completed_turns=2)
        # Triggers when completed_turns >= after_turns
        assert chaos.should_trigger(1, completed_turns=3)
        assert chaos.should_trigger(1, completed_turns=4)

    def test_between_turns(self) -> None:
        # between_turns triggers when completed_turns >= after_turn AND current_turn == 0
        chaos = ToolTimeoutChaos(between_turns=(2, 4))
        # current_turn must be 0 (between turns, not during a turn)
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=2)
        assert not chaos.should_trigger(1, current_turn=0, completed_turns=1)
        # Triggers when completed_turns >= 2 and current_turn == 0
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=3)


class TestToolChaosProviderFilter:
    """Tests for provider filtering on tool chaos."""

    def test_provider_filter_matches(self) -> None:
        chaos = ToolErrorChaos(provider="anthropic", always=True)
        assert chaos.should_trigger(1, provider="anthropic")

    def test_provider_filter_no_match(self) -> None:
        chaos = ToolErrorChaos(provider="anthropic", always=True)
        assert not chaos.should_trigger(1, provider="openai")

    def test_no_provider_filter_allows_all(self) -> None:
        chaos = ToolErrorChaos(always=True)
        assert chaos.should_trigger(1, provider="anthropic")
        assert chaos.should_trigger(1, provider="openai")
        assert chaos.should_trigger(1, provider="any")
