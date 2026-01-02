"""Tests for integrations/pydantic_evals.py - Pydantic Evals integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_chaos.core.context import ChaosContext
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder
from agent_chaos.scenario.model import TurnResult
from agent_chaos.integrations.pydantic_evals import (
    PydanticEvalsAssertion,
    as_assertion,
    build_evaluator_context,
    _parse_evaluator_output,
    _format_conversation_for_eval,
    _extract_chaos_context,
)


@pytest.fixture
def ctx() -> ChaosContext:
    """Fresh ChaosContext for testing."""
    return ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        recorder=Recorder(metrics=MetricsStore()),
        session_id="test-123",
    )


@pytest.fixture
def ctx_with_turns() -> ChaosContext:
    """ChaosContext with turn results."""
    ctx = ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        recorder=Recorder(metrics=MetricsStore()),
        session_id="test-123",
    )
    ctx.turn_results = [
        TurnResult(
            turn_number=1,
            input="What's the weather?",
            success=True,
            response="It's sunny today!",
            duration_s=1.5,
            llm_calls=2,
        ),
        TurnResult(
            turn_number=2,
            input="Thanks!",
            success=True,
            response="You're welcome!",
            duration_s=0.8,
            llm_calls=1,
        ),
    ]
    ctx.elapsed_s = 2.3
    return ctx


@pytest.fixture
def ctx_single_turn() -> ChaosContext:
    """ChaosContext with single turn result."""
    ctx = ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        recorder=Recorder(metrics=MetricsStore()),
        session_id="test-123",
    )
    ctx.agent_input = "Hello"
    ctx.agent_output = "Hi there!"
    ctx.elapsed_s = 1.0
    return ctx


@dataclass
class MockBoolEvaluator:
    """Mock evaluator that returns a boolean."""

    result: bool = True
    name: str = "MockBool"

    def evaluate_sync(self, ctx: Any) -> bool:
        return self.result


@dataclass
class MockScoreEvaluator:
    """Mock evaluator that returns a float score."""

    score: float = 0.8
    name: str = "MockScore"

    def evaluate_sync(self, ctx: Any) -> float:
        return self.score


@dataclass
class MockReasonEvaluator:
    """Mock evaluator that returns an EvaluationReason."""

    value: bool | float = True
    reason: str = "Looks good"
    name: str = "MockReason"

    def evaluate_sync(self, ctx: Any) -> Any:
        from pydantic_evals.evaluators.evaluator import EvaluationReason

        return EvaluationReason(value=self.value, reason=self.reason)


@dataclass
class MockDictEvaluator:
    """Mock evaluator that returns a dict of results."""

    results: dict[str, Any] | None = None
    name: str = "MockDict"

    def evaluate_sync(self, ctx: Any) -> dict[str, Any]:
        if self.results is not None:
            return self.results
        return {"accuracy": 0.9, "relevance": 0.8}


@dataclass
class MockFailingEvaluator:
    """Mock evaluator that raises an exception."""

    name: str = "MockFailing"

    def evaluate_sync(self, ctx: Any) -> bool:
        raise ValueError("Evaluation failed!")


class TestFormatConversationForEval:
    """Tests for _format_conversation_for_eval helper."""

    def test_single_turn_uses_agent_fields(self, ctx_single_turn: ChaosContext) -> None:
        input_text, output_text = _format_conversation_for_eval(ctx_single_turn)
        assert input_text == "Hello"
        assert output_text == "Hi there!"

    def test_multi_turn_formats_conversation(self, ctx_with_turns: ChaosContext) -> None:
        input_text, output_text = _format_conversation_for_eval(ctx_with_turns)
        assert "[Turn 1] User: What's the weather?" in input_text
        assert "[Turn 2] User: Thanks!" in input_text
        assert "[Turn 1] Assistant: It's sunny today!" in output_text
        assert "[Turn 2] Assistant: You're welcome!" in output_text


class TestExtractChaosContext:
    """Tests for _extract_chaos_context helper."""

    def test_no_chaos_returns_none(self, ctx: ChaosContext) -> None:
        result = _extract_chaos_context(ctx)
        assert result is None

    def test_with_chaos_events(self, ctx: ChaosContext) -> None:
        ctx.metrics.conv.entries = [
            {"type": "chaos", "fault_type": "rate_limit", "chaos_point": "llm_call", "turn_number": 1},
            {"type": "chaos", "fault_type": "tool_error", "target_tool": "get_weather", "turn_number": 2},
        ]
        result = _extract_chaos_context(ctx)
        assert result is not None
        assert "rate_limit" in result
        assert "tool_error" in result
        assert "get_weather" in result

    def test_filters_by_turn(self, ctx: ChaosContext) -> None:
        ctx.metrics.conv.entries = [
            {"type": "chaos", "fault_type": "rate_limit", "chaos_point": "llm_call", "turn_number": 1},
            {"type": "chaos", "fault_type": "timeout", "chaos_point": "llm_call", "turn_number": 3},
        ]
        result = _extract_chaos_context(ctx, turn=2)
        assert result is not None
        assert "rate_limit" in result
        assert "timeout" not in result


class TestBuildEvaluatorContext:
    """Tests for build_evaluator_context function."""

    def test_builds_context_for_single_turn(self, ctx_single_turn: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_single_turn)
        assert eval_ctx.inputs == "Hello"
        assert eval_ctx.output == "Hi there!"
        assert eval_ctx.duration == 1.0
        assert eval_ctx.name == "test-ctx"

    def test_builds_context_for_multi_turn(self, ctx_with_turns: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_with_turns)
        assert "[Turn 1]" in eval_ctx.inputs
        assert "[Turn 2]" in eval_ctx.inputs
        assert "sunny" in eval_ctx.output
        assert eval_ctx.duration == 2.3

    def test_builds_context_for_specific_turn(self, ctx_with_turns: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_with_turns, turn=1)
        assert eval_ctx.inputs == "What's the weather?"
        assert eval_ctx.output == "It's sunny today!"
        assert eval_ctx.duration == 1.5

    def test_includes_expected_output(self, ctx_single_turn: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_single_turn, expected_output="Expected response")
        assert eval_ctx.expected_output == "Expected response"

    def test_includes_chaos_info_in_metadata(self, ctx_single_turn: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_single_turn, include_chaos_info=True)
        assert eval_ctx.metadata is not None
        assert "chaos_context" in eval_ctx.metadata
        assert "No errors were injected" in eval_ctx.metadata["chaos_context"]

    def test_excludes_chaos_info_when_disabled(self, ctx_single_turn: ChaosContext) -> None:
        eval_ctx = build_evaluator_context(ctx_single_turn, include_chaos_info=False)
        assert eval_ctx.metadata is None or "chaos_context" not in (eval_ctx.metadata or {})

    def test_raises_for_invalid_turn(self, ctx_with_turns: ChaosContext) -> None:
        with pytest.raises(ValueError, match="Turn 99 not found"):
            build_evaluator_context(ctx_with_turns, turn=99)


class TestParseEvaluatorOutput:
    """Tests for _parse_evaluator_output function."""

    def test_parses_bool_true(self) -> None:
        passed, score, message = _parse_evaluator_output(True, None, "test")
        assert passed is True
        assert score is None
        assert "passed" in message

    def test_parses_bool_false(self) -> None:
        passed, score, message = _parse_evaluator_output(False, None, "test")
        assert passed is False
        assert score is None
        assert "failed" in message

    def test_parses_float_with_threshold(self) -> None:
        passed, score, message = _parse_evaluator_output(0.8, 0.7, "test")
        assert passed is True
        assert score == 0.8
        assert "0.80" in message
        assert "0.70" in message

    def test_parses_float_below_threshold(self) -> None:
        passed, score, message = _parse_evaluator_output(0.5, 0.7, "test")
        assert passed is False
        assert score == 0.5

    def test_parses_float_without_threshold(self) -> None:
        passed, score, message = _parse_evaluator_output(0.8, None, "test")
        assert passed is True  # Default threshold is 0.5
        assert score == 0.8

    def test_parses_string(self) -> None:
        passed, score, message = _parse_evaluator_output("All good", None, "test")
        assert passed is True
        assert score is None
        assert message == "All good"

    def test_parses_evaluation_reason_bool(self) -> None:
        from pydantic_evals.evaluators.evaluator import EvaluationReason

        reason = EvaluationReason(value=True, reason="Looks great")
        passed, score, message = _parse_evaluator_output(reason, None, "test")
        assert passed is True
        assert score is None
        assert "Looks great" in message

    def test_parses_evaluation_reason_score(self) -> None:
        from pydantic_evals.evaluators.evaluator import EvaluationReason

        reason = EvaluationReason(value=0.9, reason="Almost perfect")
        passed, score, message = _parse_evaluator_output(reason, 0.7, "test")
        assert passed is True
        assert score == 0.9
        assert "Almost perfect" in message

    def test_parses_dict_output(self) -> None:
        output = {"accuracy": 0.9, "relevance": 0.6}
        passed, score, message = _parse_evaluator_output(output, 0.7, "test")
        assert passed is False  # relevance 0.6 < 0.7
        assert score == 0.75  # Average of 0.9 and 0.6


class TestPydanticEvalsAssertion:
    """Tests for PydanticEvalsAssertion class."""

    def test_bool_evaluator_passes(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(evaluator=MockBoolEvaluator(result=True))
        result = assertion(ctx_single_turn)
        assert result.passed is True
        assert result.name == "pydantic-evals:MockBoolEvaluator"

    def test_bool_evaluator_fails(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(evaluator=MockBoolEvaluator(result=False))
        result = assertion(ctx_single_turn)
        assert result.passed is False

    def test_score_evaluator_with_threshold(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockScoreEvaluator(score=0.8),
            threshold=0.7,
        )
        result = assertion(ctx_single_turn)
        assert result.passed is True
        assert result.measured == 0.8
        assert result.expected == 0.7

    def test_score_evaluator_below_threshold(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockScoreEvaluator(score=0.5),
            threshold=0.7,
        )
        result = assertion(ctx_single_turn)
        assert result.passed is False

    def test_reason_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockReasonEvaluator(value=True, reason="Excellent response")
        )
        result = assertion(ctx_single_turn)
        assert result.passed is True
        assert "Excellent response" in result.message

    def test_dict_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockDictEvaluator(results={"a": 0.9, "b": 0.8}),
            threshold=0.7,
        )
        result = assertion(ctx_single_turn)
        assert result.passed is True

    def test_failing_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(evaluator=MockFailingEvaluator())
        result = assertion(ctx_single_turn)
        assert result.passed is False
        assert "Evaluation failed!" in result.message

    def test_custom_name(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockBoolEvaluator(),
            name="custom-assertion-name",
        )
        result = assertion(ctx_single_turn)
        assert result.name == "custom-assertion-name"

    def test_evaluates_specific_turn(self, ctx_with_turns: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockBoolEvaluator(),
            turn=1,
        )
        result = assertion(ctx_with_turns)
        assert result.passed is True

    def test_turn_number_arg_overrides_turn(self, ctx_with_turns: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockBoolEvaluator(),
            turn=1,
        )
        # turn_number arg (from runner) should override self.turn
        result = assertion(ctx_with_turns, turn_number=2)
        assert result.passed is True

    def test_include_chaos_info_default_true(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(evaluator=MockBoolEvaluator())
        assert assertion.include_chaos_info is True

    def test_expected_output_passed_to_context(self, ctx_single_turn: ChaosContext) -> None:
        assertion = PydanticEvalsAssertion(
            evaluator=MockBoolEvaluator(),
            expected_output="Expected value",
        )
        result = assertion(ctx_single_turn)
        assert result.passed is True


class TestAsAssertion:
    """Tests for as_assertion convenience function."""

    def test_wraps_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        assertion = as_assertion(MockBoolEvaluator(result=True))
        assert isinstance(assertion, PydanticEvalsAssertion)
        result = assertion(ctx_single_turn)
        assert result.passed is True

    def test_passes_threshold(self, ctx_single_turn: ChaosContext) -> None:
        assertion = as_assertion(MockScoreEvaluator(score=0.8), threshold=0.7)
        assert assertion.threshold == 0.7
        result = assertion(ctx_single_turn)
        assert result.passed is True

    def test_passes_expected_output(self) -> None:
        assertion = as_assertion(MockBoolEvaluator(), expected_output="Expected")
        assert assertion.expected_output == "Expected"

    def test_passes_turn(self) -> None:
        assertion = as_assertion(MockBoolEvaluator(), turn=2)
        assert assertion.turn == 2

    def test_passes_name(self) -> None:
        assertion = as_assertion(MockBoolEvaluator(), name="my-assertion")
        assert assertion.name == "my-assertion"

    def test_passes_include_chaos_info(self) -> None:
        assertion = as_assertion(MockBoolEvaluator(), include_chaos_info=False)
        assert assertion.include_chaos_info is False


class TestIntegrationWithRealEvaluators:
    """Integration tests with real pydantic-evals evaluators (no LLM calls)."""

    def test_with_custom_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        from pydantic_evals.evaluators import Evaluator
        from pydantic_evals.evaluators.context import EvaluatorContext

        @dataclass
        class ContainsHello(Evaluator[str, str, None]):
            """Custom evaluator that checks if output contains 'hello'."""

            def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
                return "hi" in ctx.output.lower()

        assertion = as_assertion(ContainsHello())
        result = assertion(ctx_single_turn)
        assert result.passed is True
        assert "pydantic-evals:ContainsHello" in result.name

    def test_with_score_returning_evaluator(self, ctx_single_turn: ChaosContext) -> None:
        from pydantic_evals.evaluators import Evaluator
        from pydantic_evals.evaluators.context import EvaluatorContext

        @dataclass
        class LengthScore(Evaluator[str, str, None]):
            """Custom evaluator that scores based on response length."""

            min_length: int = 5

            def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> float:
                length = len(ctx.output)
                return min(1.0, length / (self.min_length * 2))

        assertion = as_assertion(LengthScore(min_length=5), threshold=0.5)
        result = assertion(ctx_single_turn)
        assert result.passed is True
        assert result.measured is not None
        assert result.measured > 0
