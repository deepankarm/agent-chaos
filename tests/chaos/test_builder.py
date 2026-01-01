"""Tests for ChaosBuilder fluent API."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.chaos.llm import RateLimitChaos, llm_rate_limit
from agent_chaos.chaos.stream import StreamCutChaos, llm_stream_cut
from agent_chaos.chaos.tool import ToolErrorChaos, tool_error
from agent_chaos.types import ChaosAction


class TestChaosBuilderCallTriggers:
    """Tests for call-based trigger configuration."""

    def test_on_call(self) -> None:
        chaos = llm_rate_limit().on_call(3).build()
        assert chaos.on_call == 3
        assert not chaos.should_trigger(1)
        assert not chaos.should_trigger(2)
        assert chaos.should_trigger(3)

    def test_after_calls(self) -> None:
        chaos = llm_rate_limit().after_calls(2).build()
        assert chaos.after_calls == 2
        assert not chaos.should_trigger(1)
        assert not chaos.should_trigger(2)
        assert chaos.should_trigger(3)
        assert chaos.should_trigger(4)


class TestChaosBuilderTurnTriggers:
    """Tests for turn-based trigger configuration."""

    def test_on_turn(self) -> None:
        chaos = llm_rate_limit().on_turn(2).build()
        assert chaos.on_turn == 2
        assert not chaos.should_trigger(1, current_turn=1)
        assert chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(1, current_turn=3)

    def test_after_turns(self) -> None:
        chaos = llm_rate_limit().after_turns(2).build()
        assert chaos.after_turns == 2
        assert not chaos.should_trigger(1, completed_turns=1)
        assert chaos.should_trigger(1, completed_turns=2)
        assert chaos.should_trigger(1, completed_turns=3)

    def test_between_turns(self) -> None:
        chaos = llm_rate_limit().between_turns(2, 4).build()
        assert chaos.between_turns == (2, 4)
        # Between turns triggers when current_turn == 0 and completed_turns >= after
        assert not chaos.should_trigger(1, current_turn=0, completed_turns=1)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=3)


class TestChaosBuilderOtherTriggers:
    """Tests for other trigger configurations."""

    def test_with_probability(self) -> None:
        chaos = llm_rate_limit().with_probability(0.5).build()
        assert chaos.probability == 0.5

    def test_for_provider(self) -> None:
        chaos = llm_rate_limit().for_provider("anthropic").build()
        assert chaos.provider == "anthropic"

    def test_for_tool(self) -> None:
        chaos = tool_error().for_tool("weather").build()
        assert chaos.tool_name == "weather"

    def test_always(self) -> None:
        chaos = llm_rate_limit().always().build()
        assert chaos.always is True
        assert chaos.should_trigger(1)
        assert chaos.should_trigger(100)


class TestChaosBuilderChaining:
    """Tests for chaining multiple configuration methods."""

    def test_chain_on_turn_and_on_call(self) -> None:
        chaos = llm_rate_limit().on_turn(2).on_call(3).build()
        assert chaos.on_turn == 2
        assert chaos.on_call == 3
        # Must be on turn 2 AND call 3
        assert not chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(3, current_turn=1)
        assert chaos.should_trigger(3, current_turn=2)

    def test_chain_provider_and_call(self) -> None:
        chaos = llm_rate_limit().for_provider("anthropic").on_call(1).build()
        assert chaos.provider == "anthropic"
        assert chaos.on_call == 1
        assert chaos.should_trigger(1, provider="anthropic")
        assert not chaos.should_trigger(1, provider="openai")

    def test_chain_multiple_configs(self) -> None:
        chaos = (
            llm_rate_limit()
            .for_provider("anthropic")
            .on_turn(2)
            .after_calls(1)
            .build()
        )
        assert chaos.provider == "anthropic"
        assert chaos.on_turn == 2
        assert chaos.after_calls == 1


class TestChaosBuilderBuild:
    """Tests for build() method."""

    def test_build_returns_correct_type(self) -> None:
        chaos = llm_rate_limit().on_call(1).build()
        assert isinstance(chaos, RateLimitChaos)

    def test_build_preserves_defaults(self) -> None:
        chaos = llm_rate_limit(45.0).on_call(1).build()
        assert chaos.retry_after == 45.0

    def test_build_multiple_times_creates_new_instances(self) -> None:
        builder = llm_rate_limit().on_call(1)
        chaos1 = builder.build()
        chaos2 = builder.build()
        assert chaos1 is not chaos2


class TestChaosBuilderPassthrough:
    """Tests for pass-through methods that delegate to built chaos."""

    def test_point_property_passthrough(self) -> None:
        builder = llm_rate_limit().on_call(1)
        # point property should work on builder
        assert builder.point == ChaosPoint.LLM_CALL

    def test_should_trigger_passthrough(self) -> None:
        builder = llm_rate_limit().on_call(2)
        # should_trigger should work on builder
        assert not builder.should_trigger(1)
        assert builder.should_trigger(2)

    def test_apply_passthrough(self) -> None:
        builder = llm_rate_limit().on_call(1)
        # apply should work on builder
        result = builder.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None


class TestChaosBuilderDefaults:
    """Tests for default values in factory functions."""

    def test_llm_rate_limit_defaults(self) -> None:
        chaos = llm_rate_limit().on_call(1).build()
        assert chaos.retry_after == 30.0  # default

    def test_llm_rate_limit_custom_retry(self) -> None:
        chaos = llm_rate_limit(60.0).on_call(1).build()
        assert chaos.retry_after == 60.0

    def test_llm_stream_cut_defaults(self) -> None:
        chaos = llm_stream_cut(5).on_call(1).build()
        assert chaos.after_chunks == 5

    def test_tool_error_defaults(self) -> None:
        chaos = tool_error().on_call(1).build()
        assert chaos.error_message == "Tool error"

    def test_tool_error_custom_message(self) -> None:
        chaos = tool_error("Custom error").on_call(1).build()
        assert chaos.error_message == "Custom error"
