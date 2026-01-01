"""Tests for scenario/assertions.py - Assertion classes."""

from __future__ import annotations

import pytest

from agent_chaos.core.context import ChaosContext
from agent_chaos.scenario.model import TurnResult
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.scenario.assertions import (
    AllTurnsComplete,
    AssertionResult,
    CompletesWithin,
    ExpectError,
    MaxFailedCalls,
    MaxInputTokens,
    MaxInputTokensPerCall,
    MaxLLMCalls,
    MaxOutputTokens,
    MaxOutputTokensPerCall,
    MaxTokens,
    MaxTokensPerCall,
    MaxTokensPerTurn,
    MaxTotalLLMCalls,
    MinChaosInjected,
    MinLLMCalls,
    RecoveredAfterFailure,
    TurnCompletes,
    TurnCompletesWithin,
    TurnMaxLLMCalls,
    TurnResponseContains,
)


@pytest.fixture
def ctx() -> ChaosContext:
    """Fresh ChaosContext for testing."""
    return ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        metrics=MetricsStore(),
        session_id="test-123",
    )


@pytest.fixture
def ctx_with_turns() -> ChaosContext:
    """ChaosContext with turn results."""
    ctx = ChaosContext(
        name="test-ctx",
        injector=ChaosInjector(chaos=[]),
        metrics=MetricsStore(),
        session_id="test-123",
    )
    ctx.turn_results = [
        TurnResult(turn_number=1, input="Hi", success=True, response="Hello!", duration_s=1.0, llm_calls=2),
        TurnResult(turn_number=2, input="More", success=False, response="", duration_s=0.5, error="Rate limit", llm_calls=3),
        TurnResult(turn_number=3, input="Retry", success=True, response="I recovered!", duration_s=2.0, llm_calls=1),
    ]
    return ctx


@pytest.fixture
def ctx_with_history(ctx: ChaosContext) -> ChaosContext:
    """ChaosContext with call history."""
    ctx.metrics.call_history = [
        {"success": True, "latency": 0.5, "usage": {"input_tokens": 100, "output_tokens": 50}},
        {"success": True, "latency": 1.0, "usage": {"input_tokens": 200, "output_tokens": 100}},
        {"success": False, "latency": 0.2, "usage": {"input_tokens": 50, "output_tokens": 0}},
    ]
    ctx.metrics.call_count = 3
    ctx.metrics.latencies = [0.5, 1.0]
    ctx.metrics._ttft_times = [0.1, 0.2]
    return ctx


class TestAssertionResult:
    """Tests for AssertionResult dataclass."""

    def test_basic_creation(self) -> None:
        result = AssertionResult(name="test", passed=True)
        assert result.name == "test"
        assert result.passed is True
        assert result.message == ""

    def test_with_message(self) -> None:
        result = AssertionResult(name="test", passed=False, message="Something failed")
        assert result.passed is False
        assert result.message == "Something failed"

    def test_with_measured_and_expected(self) -> None:
        result = AssertionResult(
            name="test",
            passed=True,
            measured=5,
            expected=10,
        )
        assert result.measured == 5
        assert result.expected == 10


class TestCompletesWithin:
    """Tests for CompletesWithin assertion."""

    def test_passes_when_within_timeout(self, ctx: ChaosContext) -> None:
        ctx.elapsed_s = 5.0
        assertion = CompletesWithin(timeout_s=10.0)
        result = assertion(ctx)
        assert result.passed is True
        assert "5.00s" in result.message
        assert result.measured == 5.0
        assert result.expected == 10.0

    def test_fails_when_exceeds_timeout(self, ctx: ChaosContext) -> None:
        ctx.elapsed_s = 15.0
        assertion = CompletesWithin(timeout_s=10.0)
        result = assertion(ctx)
        assert result.passed is False
        assert "timeout" in result.message

    def test_passes_at_exactly_timeout(self, ctx: ChaosContext) -> None:
        ctx.elapsed_s = 10.0
        assertion = CompletesWithin(timeout_s=10.0)
        result = assertion(ctx)
        assert result.passed is True

    def test_custom_name(self, ctx: ChaosContext) -> None:
        ctx.elapsed_s = 1.0
        assertion = CompletesWithin(timeout_s=10.0, name="custom_timeout")
        result = assertion(ctx)
        assert result.name == "custom_timeout"


class TestMaxLLMCalls:
    """Tests for MaxLLMCalls assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MaxLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is True
        assert "llm_calls=5" in result.message

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 15
        assertion = MaxLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is False

    def test_passes_at_exactly_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 10
        assertion = MaxLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is True

    def test_custom_name(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MaxLLMCalls(max_calls=10, name="call_limit")
        result = assertion(ctx)
        assert result.name == "call_limit"

    def test_includes_values(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MaxLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.measured == 5
        assert result.expected == 10


class TestMinLLMCalls:
    """Tests for MinLLMCalls assertion."""

    def test_passes_when_above_minimum(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MinLLMCalls(min_calls=3)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_below_minimum(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 2
        assertion = MinLLMCalls(min_calls=5)
        result = assertion(ctx)
        assert result.passed is False

    def test_custom_name(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MinLLMCalls(min_calls=3, name="min_call_check")
        result = assertion(ctx)
        assert result.name == "min_call_check"

    def test_includes_values(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MinLLMCalls(min_calls=3)
        result = assertion(ctx)
        assert result.measured == 5
        assert result.expected == 3


class TestMinChaosInjected:
    """Tests for MinChaosInjected assertion."""

    def test_passes_when_above_minimum(self, ctx: ChaosContext) -> None:
        ctx.metrics.faults_injected = [("c1", "e1"), ("c2", "e2"), ("c3", "e3")]
        assertion = MinChaosInjected(min_chaos=2)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_below_minimum(self, ctx: ChaosContext) -> None:
        ctx.metrics.faults_injected = [("c1", "e1")]
        assertion = MinChaosInjected(min_chaos=3)
        result = assertion(ctx)
        assert result.passed is False

    def test_passes_at_exactly_minimum(self, ctx: ChaosContext) -> None:
        ctx.metrics.faults_injected = [("c1", "e1"), ("c2", "e2")]
        assertion = MinChaosInjected(min_chaos=2)
        result = assertion(ctx)
        assert result.passed is True


class TestMaxFailedCalls:
    """Tests for MaxFailedCalls assertion."""

    def test_passes_when_no_failures(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"success": True},
            {"success": True},
        ]
        assertion = MaxFailedCalls(max_failed=0)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_too_many_failures(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"success": False},
            {"success": False},
            {"success": True},
        ]
        assertion = MaxFailedCalls(max_failed=1)
        result = assertion(ctx)
        assert result.passed is False

    def test_passes_at_exactly_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"success": True},
            {"success": False},
            {"success": True},
        ]
        assertion = MaxFailedCalls(max_failed=1)
        result = assertion(ctx)
        assert result.passed is True

    def test_with_all_failures(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"success": False},
            {"success": False},
            {"success": False},
        ]
        assertion = MaxFailedCalls(max_failed=2)
        result = assertion(ctx)
        assert result.passed is False


class TestExpectError:
    """Tests for ExpectError assertion."""

    def test_passes_when_error_matches(self, ctx: ChaosContext) -> None:
        ctx.error = "Rate limit exceeded"
        assertion = ExpectError(pattern="(?i)rate limit")  # case-insensitive pattern
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_no_error(self, ctx: ChaosContext) -> None:
        ctx.error = None
        assertion = ExpectError(pattern="rate limit")
        result = assertion(ctx)
        assert result.passed is False
        assert "expected error" in result.message

    def test_fails_when_error_no_match(self, ctx: ChaosContext) -> None:
        ctx.error = "Timeout error"
        assertion = ExpectError(pattern="rate limit")
        result = assertion(ctx)
        assert result.passed is False

    def test_allows_error_flag(self) -> None:
        assertion = ExpectError(pattern=".*")
        assert assertion.allows_error is True


class TestTurnCompletes:
    """Tests for TurnCompletes assertion."""

    def test_passes_when_turn_succeeds(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=1)
        result = assertion(ctx_with_turns)
        assert result.passed is True
        assert "turn 1 completed" in result.message

    def test_fails_when_turn_fails(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=2)
        result = assertion(ctx_with_turns)
        assert result.passed is False
        assert "failed" in result.message

    def test_expect_error_passes_on_failure(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=2, expect_error=True)
        result = assertion(ctx_with_turns)
        assert result.passed is True
        assert "failed as expected" in result.message

    def test_no_turn_number_fails(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=None)
        result = assertion(ctx_with_turns, turn_number=None)
        assert result.passed is False
        assert "no turn number" in result.message

    def test_turn_not_found(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=99)
        result = assertion(ctx_with_turns)
        assert result.passed is False
        assert "not found" in result.message

    def test_with_turn_number_arg(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletes(turn=None)
        result = assertion(ctx_with_turns, turn_number=2)
        assert result.passed is False  # Turn 2 failed


class TestTurnCompletesWithin:
    """Tests for TurnCompletesWithin assertion."""

    def test_passes_when_within_timeout(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletesWithin(timeout_s=5.0, turn=1)
        result = assertion(ctx_with_turns)
        assert result.passed is True
        assert "1.00s" in result.message

    def test_fails_when_exceeds_timeout(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletesWithin(timeout_s=0.5, turn=3)
        result = assertion(ctx_with_turns)
        assert result.passed is False
        assert "timeout" in result.message

    def test_with_turn_number_arg(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnCompletesWithin(timeout_s=3.0, turn=None)
        result = assertion(ctx_with_turns, turn_number=1)
        assert result.passed is True


class TestTurnResponseContains:
    """Tests for TurnResponseContains assertion."""

    def test_passes_when_substring_found(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnResponseContains(substring="Hello", turn=1)
        result = assertion(ctx_with_turns)
        assert result.passed is True

    def test_fails_when_substring_missing(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnResponseContains(substring="Goodbye", turn=1)
        result = assertion(ctx_with_turns)
        assert result.passed is False

    def test_case_insensitive_match(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnResponseContains(substring="HELLO", turn=1, case_sensitive=False)
        result = assertion(ctx_with_turns)
        assert result.passed is True

    def test_case_sensitive_match_fails(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnResponseContains(substring="HELLO", turn=1, case_sensitive=True)
        result = assertion(ctx_with_turns)
        assert result.passed is False

    def test_with_turn_number_arg(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnResponseContains(substring="recovered", turn=None)
        result = assertion(ctx_with_turns, turn_number=3)
        assert result.passed is True


class TestTurnMaxLLMCalls:
    """Tests for TurnMaxLLMCalls assertion."""

    def test_passes_when_within_limit(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnMaxLLMCalls(max_calls=5, turn=1)
        result = assertion(ctx_with_turns)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnMaxLLMCalls(max_calls=1, turn=2)
        result = assertion(ctx_with_turns)
        assert result.passed is False

    def test_with_turn_number_arg(self, ctx_with_turns: ChaosContext) -> None:
        assertion = TurnMaxLLMCalls(max_calls=5, turn=None)
        result = assertion(ctx_with_turns, turn_number=2)
        assert result.passed is True


class TestAllTurnsComplete:
    """Tests for AllTurnsComplete assertion."""

    def test_passes_when_all_succeed(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=True, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=2, input="B", success=True, response="", duration_s=1.0, llm_calls=1),
        ]
        assertion = AllTurnsComplete()
        result = assertion(ctx)
        assert result.passed is True
        assert "successfully" in result.message

    def test_fails_when_turn_fails(self, ctx_with_turns: ChaosContext) -> None:
        assertion = AllTurnsComplete()
        result = assertion(ctx_with_turns)
        assert result.passed is False
        assert "2" in result.message  # Turn 2 failed

    def test_passes_with_allowed_failures(self, ctx_with_turns: ChaosContext) -> None:
        assertion = AllTurnsComplete(allow_failures=1)
        result = assertion(ctx_with_turns)
        assert result.passed is True

    def test_no_turns_passes(self, ctx: ChaosContext) -> None:
        assertion = AllTurnsComplete()
        result = assertion(ctx)
        assert result.passed is True
        assert "legacy" in result.message

    def test_with_multiple_allowed_failures(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=True, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=2, input="B", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=3, input="C", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=4, input="D", success=True, response="", duration_s=1.0, llm_calls=1),
        ]
        assertion = AllTurnsComplete(allow_failures=2)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_allowed_failures(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=2, input="B", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=3, input="C", success=False, response="", duration_s=1.0, llm_calls=1),
        ]
        assertion = AllTurnsComplete(allow_failures=2)
        result = assertion(ctx)
        assert result.passed is False


class TestRecoveredAfterFailure:
    """Tests for RecoveredAfterFailure assertion."""

    def test_passes_when_recovered(self, ctx_with_turns: ChaosContext) -> None:
        assertion = RecoveredAfterFailure(failed_turn=2)
        result = assertion(ctx_with_turns)
        assert result.passed is True
        assert "recovered on turn 3" in result.message

    def test_fails_when_no_recovery(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=True, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=2, input="B", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=3, input="C", success=False, response="", duration_s=1.0, llm_calls=1),
        ]
        assertion = RecoveredAfterFailure(failed_turn=2)
        result = assertion(ctx)
        assert result.passed is False
        assert "did not recover" in result.message

    def test_fails_when_expected_failure_succeeded(self, ctx_with_turns: ChaosContext) -> None:
        assertion = RecoveredAfterFailure(failed_turn=1)  # Turn 1 succeeded
        result = assertion(ctx_with_turns)
        assert result.passed is False
        assert "succeeded" in result.message

    def test_immediate_recovery(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=False, response="", duration_s=1.0, llm_calls=1),
            TurnResult(turn_number=2, input="B", success=True, response="Recovered!", duration_s=1.0, llm_calls=1),
        ]
        assertion = RecoveredAfterFailure(failed_turn=1)
        result = assertion(ctx)
        assert result.passed is True
        assert "turn 2" in result.message

    def test_no_subsequent_turns(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(turn_number=1, input="A", success=False, response="", duration_s=1.0, llm_calls=1),
        ]
        assertion = RecoveredAfterFailure(failed_turn=1)
        result = assertion(ctx)
        assert result.passed is False
        assert "no turns after" in result.message


class TestMaxTotalLLMCalls:
    """Tests for MaxTotalLLMCalls assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MaxTotalLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 15
        assertion = MaxTotalLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is False

    def test_passes_at_exactly_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 10
        assertion = MaxTotalLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.passed is True

    def test_includes_values_in_result(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_count = 5
        assertion = MaxTotalLLMCalls(max_calls=10)
        result = assertion(ctx)
        assert result.measured == 5
        assert result.expected == 10


class TestMaxTokens:
    """Tests for MaxTokens assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 2000, "output_tokens": 3000}},
        ]
        assertion = MaxTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 8000, "output_tokens": 7000}},
        ]
        assertion = MaxTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is False

    def test_includes_values(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 100, "output_tokens": 50}},
        ]
        assertion = MaxTokens(max_tokens=500)
        result = assertion(ctx)
        assert result.measured == 150
        assert result.expected == 500


class TestMaxInputTokens:
    """Tests for MaxInputTokens assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 3000, "output_tokens": 2000}},
            {"usage": {"input_tokens": 2000, "output_tokens": 1000}},
        ]
        assertion = MaxInputTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 5000, "output_tokens": 1000}},
            {"usage": {"input_tokens": 6000, "output_tokens": 2000}},
        ]
        assertion = MaxInputTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is False
        assert result.measured == 11000


class TestMaxOutputTokens:
    """Tests for MaxOutputTokens assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 3000, "output_tokens": 2000}},
            {"usage": {"input_tokens": 2000, "output_tokens": 3000}},
        ]
        assertion = MaxOutputTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 1000, "output_tokens": 6000}},
            {"usage": {"input_tokens": 1000, "output_tokens": 5000}},
        ]
        assertion = MaxOutputTokens(max_tokens=10000)
        result = assertion(ctx)
        assert result.passed is False
        assert result.measured == 11000


class TestMaxTokensPerCall:
    """Tests for MaxTokensPerCall assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 500, "output_tokens": 500}},
            {"usage": {"input_tokens": 400, "output_tokens": 400}},
        ]
        assertion = MaxTokensPerCall(max_tokens=2000)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 500, "output_tokens": 500}},
            {"usage": {"input_tokens": 1500, "output_tokens": 1500}},  # 3000 total exceeds 2000
        ]
        assertion = MaxTokensPerCall(max_tokens=2000)
        result = assertion(ctx)
        assert result.passed is False


class TestMaxInputTokensPerCall:
    """Tests for MaxInputTokensPerCall assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 100}},
            {"usage": {"input_tokens": 200}},
        ]
        assertion = MaxInputTokensPerCall(max_tokens=500)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"input_tokens": 100}},
            {"usage": {"input_tokens": 1000}},
        ]
        assertion = MaxInputTokensPerCall(max_tokens=500)
        result = assertion(ctx)
        assert result.passed is False


class TestMaxOutputTokensPerCall:
    """Tests for MaxOutputTokensPerCall assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"output_tokens": 100}},
            {"usage": {"output_tokens": 200}},
        ]
        assertion = MaxOutputTokensPerCall(max_tokens=500)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.metrics.call_history = [
            {"usage": {"output_tokens": 100}},
            {"usage": {"output_tokens": 1000}},
        ]
        assertion = MaxOutputTokensPerCall(max_tokens=500)
        result = assertion(ctx)
        assert result.passed is False


class TestMaxTokensPerTurn:
    """Tests for MaxTokensPerTurn assertion."""

    def test_passes_when_within_limit(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(
                turn_number=1, input="Hi", success=True, response="Hello!",
                duration_s=1.0, llm_calls=1, input_tokens=100, output_tokens=50, total_tokens=150
            ),
            TurnResult(
                turn_number=2, input="More", success=True, response="Sure!",
                duration_s=1.0, llm_calls=1, input_tokens=200, output_tokens=100, total_tokens=300
            ),
        ]
        assertion = MaxTokensPerTurn(max_tokens=500, turn=2)
        result = assertion(ctx)
        assert result.passed is True

    def test_fails_when_exceeds_limit(self, ctx: ChaosContext) -> None:
        ctx.turn_results = [
            TurnResult(
                turn_number=1, input="Hi", success=True, response="Hello!",
                duration_s=1.0, llm_calls=1, input_tokens=100, output_tokens=50, total_tokens=150
            ),
            TurnResult(
                turn_number=2, input="More", success=True, response="Long response",
                duration_s=1.0, llm_calls=1, input_tokens=400, output_tokens=300, total_tokens=700
            ),
        ]
        assertion = MaxTokensPerTurn(max_tokens=500, turn=2)
        result = assertion(ctx)
        assert result.passed is False
