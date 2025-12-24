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
class MinFaultsInjected:
    """Injected faults must be >= `min_faults`."""

    min_faults: int
    name: str = "min_faults_injected"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        faults = ctx.metrics.faults_injected
        count = len(faults)
        passed = count >= self.min_faults
        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"faults_injected={count} (min {self.min_faults})",
            measured=count,
            expected=self.min_faults,
        )


@dataclass
class ExpectError:
    """Scenario is expected to raise an error matching `pattern`.

    This enables “failure-mode” scenarios to be treated as PASS when the expected
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
