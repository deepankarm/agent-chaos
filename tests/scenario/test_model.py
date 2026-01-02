"""Tests for scenario/model.py - Scenario model classes."""

from __future__ import annotations

import pytest

from agent_chaos.scenario.model import (
    at,
    Turn,
    TurnResult,
    BaselineScenario,
    ChaosScenario,
)
from agent_chaos.chaos.llm import llm_rate_limit, RateLimitChaos
from agent_chaos.chaos.tool import tool_error


def dummy_agent(ctx, turn_input: str) -> str:
    """Dummy agent for testing."""
    return f"Response to: {turn_input}"


class TestAt:
    """Tests for the 'at' helper class."""

    def test_basic_creation(self) -> None:
        target = at(turn=2)
        assert target.turn == 2
        assert target.chaos == []
        assert target.assertions == []

    def test_with_chaos(self) -> None:
        chaos = [llm_rate_limit().on_call(1)]
        target = at(turn=1, chaos=chaos)
        assert target.turn == 1
        assert len(target.chaos) == 1

    def test_with_assertions(self) -> None:
        class DummyAssertion:
            pass

        target = at(turn=3, assertions=[DummyAssertion()])
        assert len(target.assertions) == 1


class TestTurn:
    """Tests for Turn class."""

    def test_static_input(self) -> None:
        turn = Turn("What's the weather?")
        assert turn.get_input([]) == "What's the weather?"
        assert turn.is_dynamic() is False

    def test_dynamic_input(self) -> None:
        def follow_up(history: list[TurnResult]) -> str:
            if history:
                return f"Following up on turn {history[-1].turn_number}"
            return "First message"

        turn = Turn(input=follow_up)
        assert turn.is_dynamic() is True

        # Empty history
        assert turn.get_input([]) == "First message"

        # With history
        history = [
            TurnResult(
                turn_number=1,
                input="Hello",
                response="Hi!",
                success=True,
                duration_s=1.0,
                llm_calls=1,
            )
        ]
        assert turn.get_input(history) == "Following up on turn 1"

    def test_turn_with_chaos(self) -> None:
        turn = Turn(
            input="Test",
            chaos=[llm_rate_limit().on_call(1)],
        )
        assert len(turn.chaos) == 1

    def test_turn_with_assertions(self) -> None:
        class DummyAssertion:
            pass

        turn = Turn(
            input="Test",
            assertions=[DummyAssertion()],
        )
        assert len(turn.assertions) == 1


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_basic_creation(self) -> None:
        result = TurnResult(
            turn_number=1,
            input="Hello",
            response="Hi there!",
            success=True,
            duration_s=1.5,
            llm_calls=2,
        )
        assert result.turn_number == 1
        assert result.input == "Hello"
        assert result.response == "Hi there!"
        assert result.success is True
        assert result.duration_s == 1.5
        assert result.llm_calls == 2
        assert result.error is None

    def test_with_error(self) -> None:
        result = TurnResult(
            turn_number=2,
            input="Bad request",
            response="",
            success=False,
            duration_s=0.1,
            llm_calls=1,
            error="Rate limit exceeded",
        )
        assert result.success is False
        assert result.error == "Rate limit exceeded"

    def test_with_token_counts(self) -> None:
        result = TurnResult(
            turn_number=1,
            input="Hello",
            response="Hi!",
            success=True,
            duration_s=1.0,
            llm_calls=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    def test_with_chaos(self) -> None:
        result = TurnResult(
            turn_number=1,
            input="Test",
            response="Response",
            success=True,
            duration_s=1.0,
            llm_calls=1,
            chaos=[{"type": "rate_limit", "call": 1}],
        )
        assert len(result.chaos) == 1

    def test_default_values(self) -> None:
        result = TurnResult(
            turn_number=1,
            input="Hello",
            response="Hi!",
            success=True,
            duration_s=1.0,
            llm_calls=1,
        )
        assert result.error is None
        assert result.chaos == []
        assert result.assertion_results == []
        assert result.is_dynamic is False
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0


class TestBaselineScenario:
    """Tests for BaselineScenario."""

    def test_basic_creation(self) -> None:
        scenario = BaselineScenario(
            name="test-scenario",
            description="A test scenario",
            agent=dummy_agent,
            turns=[
                Turn("Hello"),
                Turn("How are you?"),
            ],
        )
        assert scenario.name == "test-scenario"
        assert scenario.description == "A test scenario"
        assert len(scenario.turns) == 2
        assert scenario.providers == ["anthropic"]
        assert scenario.assertions == []
        assert scenario.tags == []
        assert scenario.meta == {}

    def test_with_custom_providers(self) -> None:
        scenario = BaselineScenario(
            name="multi-provider",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
            providers=["anthropic", "openai"],
        )
        assert scenario.providers == ["anthropic", "openai"]

    def test_with_tags(self) -> None:
        scenario = BaselineScenario(
            name="tagged",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
            tags=["smoke", "critical"],
        )
        assert scenario.tags == ["smoke", "critical"]

    def test_with_meta(self) -> None:
        scenario = BaselineScenario(
            name="with-meta",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
            meta={"priority": 1, "owner": "team-a"},
        )
        assert scenario.meta == {"priority": 1, "owner": "team-a"}

    def test_rejects_chaos_in_turns(self) -> None:
        """Baseline scenarios cannot have chaos in turns."""
        with pytest.raises(ValueError, match="cannot have chaos"):
            BaselineScenario(
                name="bad-baseline",
                description="Test",
                agent=dummy_agent,
                turns=[
                    Turn("Hi", chaos=[llm_rate_limit().on_call(1)]),
                ],
            )


class TestBaselineScenarioVariant:
    """Tests for BaselineScenario.variant() method."""

    @pytest.fixture
    def baseline(self) -> BaselineScenario:
        return BaselineScenario(
            name="customer-journey",
            description="Standard customer flow",
            agent=dummy_agent,
            turns=[
                Turn("Check order status"),
                Turn("Request refund"),
                Turn("Confirm refund"),
            ],
            assertions=[],
            tags=["customer"],
        )

    def test_basic_variant(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(
            name="customer-journey-chaos",
        )
        assert variant.name == "customer-journey-chaos"
        assert variant.description == "Standard customer flow"
        assert variant.parent == "customer-journey"
        assert len(variant.turns) == 3

    def test_variant_with_description(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(
            name="chaos-variant",
            description="Testing under chaos",
        )
        assert variant.description == "Testing under chaos"

    def test_variant_with_global_chaos(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(
            name="rate-limited",
            chaos=[llm_rate_limit().after_calls(2)],
        )
        assert len(variant.chaos) == 1

    def test_variant_with_turn_chaos(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(
            name="tool-fails",
            turns=[
                at(1, chaos=[tool_error().for_tool("refund")]),
            ],
        )
        # Turn 1 should have chaos, others should not
        assert len(variant.turns[1].chaos) == 1
        assert len(variant.turns[0].chaos) == 0
        assert len(variant.turns[2].chaos) == 0

    def test_variant_with_turn_assertions(self, baseline: BaselineScenario) -> None:
        class DummyAssertion:
            pass

        variant = baseline.variant(
            name="with-assertions",
            turns=[
                at(2, assertions=[DummyAssertion()]),
            ],
        )
        assert len(variant.turns[2].assertions) == 1

    def test_variant_inherits_parent_tags(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(
            name="tagged-variant",
            tags=["chaos"],
        )
        assert "customer" in variant.tags
        assert "chaos" in variant.tags

    def test_variant_inherits_parent_assertions(self, baseline: BaselineScenario) -> None:
        class ParentAssertion:
            pass

        baseline_with_assertion = BaselineScenario(
            name="with-assertion",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
            assertions=[ParentAssertion()],
        )

        class ChildAssertion:
            pass

        variant = baseline_with_assertion.variant(
            name="variant",
            assertions=[ChildAssertion()],
        )
        assert len(variant.assertions) == 2

    def test_variant_inherits_providers(self, baseline: BaselineScenario) -> None:
        variant = baseline.variant(name="same-providers")
        assert variant.providers == baseline.providers


class TestChaosScenario:
    """Tests for ChaosScenario."""

    def test_basic_creation(self) -> None:
        scenario = ChaosScenario(
            name="chaos-test",
            description="Testing chaos",
            agent=dummy_agent,
            turns=[
                Turn("Hello", chaos=[llm_rate_limit().on_call(1)]),
            ],
            chaos=[llm_rate_limit().after_calls(3)],
        )
        assert scenario.name == "chaos-test"
        assert len(scenario.chaos) == 1
        assert len(scenario.turns[0].chaos) == 1
        assert scenario.parent == ""

    def test_with_parent(self) -> None:
        scenario = ChaosScenario(
            name="derived",
            description="Derived from baseline",
            agent=dummy_agent,
            turns=[Turn("Hi")],
            parent="baseline-scenario",
        )
        assert scenario.parent == "baseline-scenario"

    def test_no_variant_method(self) -> None:
        """ChaosScenario should not have a variant method."""
        scenario = ChaosScenario(
            name="chaos",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
        )
        assert not hasattr(scenario, "variant") or not callable(
            getattr(scenario, "variant", None)
        )


class TestScenarioTypeUnion:
    """Tests for Scenario type alias."""

    def test_baseline_is_scenario(self) -> None:
        from agent_chaos.scenario.model import Scenario

        baseline = BaselineScenario(
            name="test",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
        )
        # Type check - should accept both types
        scenario: Scenario = baseline
        assert isinstance(scenario, BaselineScenario)

    def test_chaos_is_scenario(self) -> None:
        from agent_chaos.scenario.model import Scenario

        chaos = ChaosScenario(
            name="test",
            description="Test",
            agent=dummy_agent,
            turns=[Turn("Hi")],
        )
        scenario: Scenario = chaos
        assert isinstance(scenario, ChaosScenario)
