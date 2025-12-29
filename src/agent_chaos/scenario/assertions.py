"""Assertion library (contracts) for scenarios.

Assertions are any callable that accepts `ctx: ChaosContext` and returns `AssertionResult | bool`.

Examples:
    # Class-based (dataclass for convenience)
    CompletesWithin(timeout_s=60.0)

    # Plain function
    def my_assertion(ctx: ChaosContext) -> AssertionResult:
        return AssertionResult(name="custom", passed=ctx.metrics.total_calls > 0)

    # Lambda
    lambda ctx: AssertionResult(name="simple", passed=ctx.error is None)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_chaos import ChaosContext


@dataclass
class AssertionResult:
    name: str
    passed: bool
    message: str = ""
    measured: Any | None = None
    expected: Any | None = None


@dataclass
class CompletesWithin:
    """Scenario must complete within `timeout_s`."""

    timeout_s: float
    name: str = "completes_within"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        elapsed_s = ctx.elapsed_s or 0.0
        passed = elapsed_s <= self.timeout_s
        msg = (
            f"completed in {elapsed_s:.2f}s (budget {self.timeout_s:.2f}s)"
            if passed
            else f"timeout: completed in {elapsed_s:.2f}s (budget {self.timeout_s:.2f}s)"
        )
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=msg,
            measured=elapsed_s,
            expected=self.timeout_s,
        )


@dataclass
class MaxLLMCalls:
    """Total LLM calls (spans) must be <= `max_calls`."""

    max_calls: int
    name: str = "max_llm_calls"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = getattr(ctx.metrics, "total_calls", 0)
        passed = total <= self.max_calls
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"llm_calls={total} (max {self.max_calls})",
            measured=total,
            expected=self.max_calls,
        )


@dataclass
class MaxFailedCalls:
    """Failed spans must be <= `max_failed`."""

    max_failed: int
    name: str = "max_failed_calls"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        history = getattr(ctx.metrics, "call_history", []) or []
        failed = sum(1 for c in history if not c.get("success", True))
        passed = failed <= self.max_failed
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"failed_calls={failed} (max {self.max_failed})",
            measured=failed,
            expected=self.max_failed,
        )


@dataclass
class MinLLMCalls:
    """Total LLM calls (spans) must be >= `min_calls`."""

    min_calls: int
    name: str = "min_llm_calls"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = getattr(ctx.metrics, "total_calls", 0)
        passed = total >= self.min_calls
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"llm_calls={total} (min {self.min_calls})",
            measured=total,
            expected=self.min_calls,
        )


@dataclass
class MinChaosInjected:
    """Injected chaos must be >= `min_chaos`."""

    min_chaos: int
    name: str = "min_chaos_injected"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        chaos_injected = ctx.metrics.faults_injected
        count = len(chaos_injected)
        passed = count >= self.min_chaos
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"chaos_injected={count} (min {self.min_chaos})",
            measured=count,
            expected=self.min_chaos,
        )


@dataclass
class ExpectError:
    """Scenario is expected to raise an error matching `pattern`.

    This enables "failure-mode" scenarios to be treated as PASS when the expected
    error occurs under chaos.
    """

    pattern: str
    name: str = "expect_error"
    allows_error: bool = True

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        if ctx.error is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message=f"expected error /{self.pattern}/ but scenario succeeded",
                measured=None,
                expected=self.pattern,
            )
        matched = re.search(self.pattern, ctx.error) is not None
        return AssertionResult(
            name=self.name,
            passed=matched,
            message=f"error={'matched' if matched else 'did_not_match'} /{self.pattern}/",
            measured=ctx.error,
            expected=self.pattern,
        )


# =============================================================================
# Turn-Aware Assertions (Multi-Turn Scenarios)
# =============================================================================


@dataclass
class TurnCompletes:
    """Assert that a specific turn completes (or fails as expected).

    Args:
        turn: Turn number to check (1-indexed). If None, checks current turn.
        expect_error: If True, turn is expected to fail.
    """

    turn: int | None = None
    expect_error: bool = False
    name: str = "turn_completes"

    def __call__(
        self, ctx: ChaosContext, turn_number: int | None = None
    ) -> AssertionResult:
        check_turn = self.turn or turn_number
        if check_turn is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message="no turn number specified",
            )

        turn_result = ctx.get_turn_result(check_turn)
        if turn_result is None:
            return AssertionResult(
                name=f"{self.name}[turn={check_turn}]",
                passed=False,
                message=f"turn {check_turn} not found",
                measured=None,
                expected=check_turn,
            )

        if self.expect_error:
            passed = not turn_result.success
            msg = (
                f"turn {check_turn} failed as expected"
                if passed
                else f"turn {check_turn} succeeded but expected failure"
            )
        else:
            passed = turn_result.success
            msg = (
                f"turn {check_turn} completed"
                if passed
                else f"turn {check_turn} failed: {turn_result.error}"
            )

        return AssertionResult(
            name=f"{self.name}[turn={check_turn}]",
            passed=passed,
            message=msg,
            measured=turn_result.success,
            expected=not self.expect_error,
        )


@dataclass
class TurnCompletesWithin:
    """Assert that a specific turn completes within a timeout.

    Args:
        turn: Turn number to check (1-indexed). If None, checks current turn.
        timeout_s: Maximum allowed duration in seconds.
    """

    timeout_s: float
    turn: int | None = None
    name: str = "turn_completes_within"

    def __call__(
        self, ctx: ChaosContext, turn_number: int | None = None
    ) -> AssertionResult:
        check_turn = self.turn or turn_number
        if check_turn is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message="no turn number specified",
            )

        turn_result = ctx.get_turn_result(check_turn)
        if turn_result is None:
            return AssertionResult(
                name=f"{self.name}[turn={check_turn}]",
                passed=False,
                message=f"turn {check_turn} not found",
            )

        passed = turn_result.duration_s <= self.timeout_s
        msg = (
            f"turn {check_turn} completed in {turn_result.duration_s:.2f}s"
            if passed
            else f"turn {check_turn} timeout: {turn_result.duration_s:.2f}s > {self.timeout_s:.2f}s"
        )

        return AssertionResult(
            name=f"{self.name}[turn={check_turn}]",
            passed=passed,
            message=msg,
            measured=turn_result.duration_s,
            expected=self.timeout_s,
        )


@dataclass
class TurnResponseContains:
    """Assert that a turn's response contains a substring.

    Args:
        turn: Turn number to check (1-indexed). If None, checks current turn.
        substring: Text that must appear in the response.
        case_sensitive: Whether to do case-sensitive matching.
    """

    substring: str
    turn: int | None = None
    case_sensitive: bool = False
    name: str = "turn_response_contains"

    def __call__(
        self, ctx: ChaosContext, turn_number: int | None = None
    ) -> AssertionResult:
        check_turn = self.turn or turn_number
        if check_turn is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message="no turn number specified",
            )

        turn_result = ctx.get_turn_result(check_turn)
        if turn_result is None:
            return AssertionResult(
                name=f"{self.name}[turn={check_turn}]",
                passed=False,
                message=f"turn {check_turn} not found",
            )

        response = turn_result.response
        search_text = self.substring

        if not self.case_sensitive:
            response = response.lower()
            search_text = search_text.lower()

        passed = search_text in response
        msg = (
            f"turn {check_turn} response contains '{self.substring}'"
            if passed
            else f"turn {check_turn} response missing '{self.substring}'"
        )

        return AssertionResult(
            name=f"{self.name}[turn={check_turn}]",
            passed=passed,
            message=msg,
            measured=turn_result.response[:100] + "..." if len(turn_result.response) > 100 else turn_result.response,
            expected=self.substring,
        )


@dataclass
class TurnMaxLLMCalls:
    """Assert that a turn uses at most N LLM calls.

    Args:
        turn: Turn number to check (1-indexed). If None, checks current turn.
        max_calls: Maximum allowed LLM calls for this turn.
    """

    max_calls: int
    turn: int | None = None
    name: str = "turn_max_llm_calls"

    def __call__(
        self, ctx: ChaosContext, turn_number: int | None = None
    ) -> AssertionResult:
        check_turn = self.turn or turn_number
        if check_turn is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message="no turn number specified",
            )

        turn_result = ctx.get_turn_result(check_turn)
        if turn_result is None:
            return AssertionResult(
                name=f"{self.name}[turn={check_turn}]",
                passed=False,
                message=f"turn {check_turn} not found",
            )

        passed = turn_result.llm_calls <= self.max_calls
        msg = f"turn {check_turn} used {turn_result.llm_calls} LLM calls (max {self.max_calls})"

        return AssertionResult(
            name=f"{self.name}[turn={check_turn}]",
            passed=passed,
            message=msg,
            measured=turn_result.llm_calls,
            expected=self.max_calls,
        )


@dataclass
class AllTurnsComplete:
    """Assert that all turns complete, with optional failure tolerance.

    Args:
        allow_failures: Number of turn failures to tolerate (default 0).
    """

    allow_failures: int = 0
    name: str = "all_turns_complete"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        if not ctx.turn_results:
            return AssertionResult(
                name=self.name,
                passed=True,
                message="no turns to check (legacy single-turn scenario)",
            )

        failed_turns = [tr for tr in ctx.turn_results if not tr.success]
        num_failed = len(failed_turns)

        passed = num_failed <= self.allow_failures
        if passed:
            if num_failed == 0:
                msg = f"all {len(ctx.turn_results)} turns completed successfully"
            else:
                msg = f"{num_failed} turn(s) failed (allowed {self.allow_failures})"
        else:
            failed_nums = [str(tr.turn_number) for tr in failed_turns]
            msg = f"turns {', '.join(failed_nums)} failed (allowed {self.allow_failures})"

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=msg,
            measured=num_failed,
            expected=self.allow_failures,
        )


@dataclass
class RecoveredAfterFailure:
    """Assert that the scenario recovered after a turn failed.

    This checks that turn N+1 (or later) succeeded after turn N failed.

    Args:
        failed_turn: The turn that was expected to fail (1-indexed).
    """

    failed_turn: int
    name: str = "recovered_after_failure"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        if not ctx.turn_results:
            return AssertionResult(
                name=self.name,
                passed=False,
                message="no turns to check",
            )

        # Check that the specified turn actually failed
        failed_result = ctx.get_turn_result(self.failed_turn)
        if failed_result is None:
            return AssertionResult(
                name=self.name,
                passed=False,
                message=f"turn {self.failed_turn} not found",
            )

        if failed_result.success:
            return AssertionResult(
                name=self.name,
                passed=False,
                message=f"turn {self.failed_turn} succeeded (expected failure)",
                measured=True,
                expected=False,
            )

        # Check that at least one subsequent turn succeeded
        subsequent_turns = [
            tr for tr in ctx.turn_results if tr.turn_number > self.failed_turn
        ]

        if not subsequent_turns:
            return AssertionResult(
                name=self.name,
                passed=False,
                message=f"no turns after {self.failed_turn} to check recovery",
            )

        recovered = any(tr.success for tr in subsequent_turns)
        if recovered:
            recovery_turn = next(tr for tr in subsequent_turns if tr.success)
            msg = f"recovered on turn {recovery_turn.turn_number} after turn {self.failed_turn} failed"
        else:
            msg = f"did not recover after turn {self.failed_turn} failed"

        return AssertionResult(
            name=self.name,
            passed=recovered,
            message=msg,
            measured=recovered,
            expected=True,
        )


@dataclass
class MaxTotalLLMCalls:
    """Assert that total LLM calls across all turns is within limit.

    Args:
        max_calls: Maximum total LLM calls allowed.
    """

    max_calls: int
    name: str = "max_total_llm_calls"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = ctx.metrics.total_calls
        passed = total <= self.max_calls

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"total LLM calls: {total} (max {self.max_calls})",
            measured=total,
            expected=self.max_calls,
        )
