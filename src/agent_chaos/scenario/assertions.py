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
        history = ctx.metrics.history or []
        failed = sum(1 for c in history if not c.success)
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
        chaos_injected = ctx.metrics.faults
        count = len(chaos_injected)
        passed = count >= self.min_chaos

        if passed:
            message = f"chaos_injected={count} (min {self.min_chaos})"
        else:
            # Build diagnostic message explaining why chaos wasn't injected
            message = self._build_diagnostic(ctx, count)

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=message,
            measured=count,
            expected=self.min_chaos,
        )

    def _build_diagnostic(self, ctx: ChaosContext, injected_count: int) -> str:
        """Build a helpful diagnostic message when chaos wasn't injected."""
        lines = [f"chaos_injected={injected_count} (min {self.min_chaos})"]

        # Get configured tool chaos
        tool_chaos_configs = []
        for chaos in ctx.injector._tool_chaos:
            tool_name = getattr(chaos, "tool_name", None) or "all"
            trigger = getattr(chaos, "_trigger", None)
            on_turn = getattr(trigger, "on_turn", None) if trigger else None
            tool_chaos_configs.append((tool_name, on_turn))

        # Get actual tool calls from conversation
        tool_calls_by_turn: dict[int, list[str]] = {}
        for entry in ctx.metrics.conv.entries:
            if entry.get("type") == "tool_call":
                turn = entry.get("turn_number", 0)
                tool = entry.get("tool_name", "unknown")
                if turn not in tool_calls_by_turn:
                    tool_calls_by_turn[turn] = []
                tool_calls_by_turn[turn].append(tool)

        if tool_chaos_configs:
            lines.append("")
            lines.append("Configured tool chaos:")
            for tool_name, on_turn in tool_chaos_configs:
                turn_info = f" on turn {on_turn} (0-indexed)" if on_turn is not None else ""
                lines.append(f"  - tool '{tool_name}'{turn_info}")

        if tool_calls_by_turn:
            lines.append("")
            lines.append("Actual tool calls:")
            for turn in sorted(tool_calls_by_turn.keys()):
                tools = tool_calls_by_turn[turn]
                lines.append(f"  Turn {turn}: {', '.join(tools)}")

        # Check for common issues
        issues = []
        for tool_name, on_turn in tool_chaos_configs:
            if tool_name != "all":
                # Check if tool was ever called
                all_tools = [t for tools in tool_calls_by_turn.values() for t in tools]
                if tool_name not in all_tools:
                    # Check for similar names
                    similar = [t for t in set(all_tools) if tool_name in t or t in tool_name]
                    if similar:
                        issues.append(f"Tool '{tool_name}' not called. Did you mean '{similar[0]}'?")
                    else:
                        issues.append(f"Tool '{tool_name}' was never called")
                elif on_turn is not None:
                    # Check if tool was called on wrong turn
                    turns_with_tool = [t for t, tools in tool_calls_by_turn.items() if tool_name in tools]
                    # Note: on_turn is 0-indexed, tool_calls_by_turn uses 1-indexed turns
                    expected_turn_1indexed = on_turn + 1
                    if expected_turn_1indexed not in turns_with_tool:
                        issues.append(
                            f"Tool '{tool_name}' called on turn(s) {turns_with_tool}, "
                            f"but chaos configured for turn {on_turn} (0-indexed = turn {expected_turn_1indexed})"
                        )

        if issues:
            lines.append("")
            lines.append("Potential issues:")
            for issue in issues:
                lines.append(f"  ⚠ {issue}")

        return "\n".join(lines)


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


# =============================================================================
# Token-Based Assertions
# =============================================================================


@dataclass
class MaxTokens:
    """Assert that total tokens across all calls is within limit.

    Args:
        max_tokens: Maximum total tokens allowed.
    """

    max_tokens: int
    name: str = "max_tokens"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = ctx.metrics.total_tokens
        passed = total <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"total tokens: {total} (max {self.max_tokens})",
            measured=total,
            expected=self.max_tokens,
        )


@dataclass
class MaxInputTokens:
    """Assert that total input tokens across all calls is within limit.

    Input tokens include user messages, system prompts, and tool results.
    Use this to detect excessive context accumulation or verbose tool outputs.

    Args:
        max_tokens: Maximum total input tokens allowed.
    """

    max_tokens: int
    name: str = "max_input_tokens"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = ctx.metrics.total_input_tokens
        passed = total <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"total input tokens: {total} (max {self.max_tokens})",
            measured=total,
            expected=self.max_tokens,
        )


@dataclass
class MaxOutputTokens:
    """Assert that total output tokens across all calls is within limit.

    Output tokens are generated by the LLM. Use this to detect overly
    verbose responses or runaway generation.

    Args:
        max_tokens: Maximum total output tokens allowed.
    """

    max_tokens: int
    name: str = "max_output_tokens"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        total = ctx.metrics.total_output_tokens
        passed = total <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"total output tokens: {total} (max {self.max_tokens})",
            measured=total,
            expected=self.max_tokens,
        )


@dataclass
class MaxTokensPerCall:
    """Assert that no single LLM call exceeds the token limit.

    Args:
        max_tokens: Maximum tokens allowed per call.
    """

    max_tokens: int
    name: str = "max_tokens_per_call"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        max_call = ctx.metrics.max_tokens_single_call
        passed = max_call <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"max tokens in single call: {max_call} (limit {self.max_tokens})",
            measured=max_call,
            expected=self.max_tokens,
        )


@dataclass
class MaxInputTokensPerCall:
    """Assert that no single LLM call exceeds the input token limit.

    Input tokens include user messages, system prompts, and tool results.
    Use this to detect when tool results are too verbose.

    Args:
        max_tokens: Maximum input tokens allowed per call.
    """

    max_tokens: int
    name: str = "max_input_tokens_per_call"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        max_call = 0
        for call in ctx.metrics.history:
            input_tok = call.usage.get("input_tokens") or 0
            if input_tok > max_call:
                max_call = input_tok

        passed = max_call <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"max input tokens in single call: {max_call} (limit {self.max_tokens})",
            measured=max_call,
            expected=self.max_tokens,
        )


@dataclass
class MaxOutputTokensPerCall:
    """Assert that no single LLM call exceeds the output token limit.

    Output tokens are generated by the LLM. Use this to detect when
    the model generates excessively long responses.

    Args:
        max_tokens: Maximum output tokens allowed per call.
    """

    max_tokens: int
    name: str = "max_output_tokens_per_call"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        max_call = 0
        for call in ctx.metrics.history:
            output_tok = call.usage.get("output_tokens") or 0
            if output_tok > max_call:
                max_call = output_tok

        passed = max_call <= self.max_tokens

        return AssertionResult(
            name=self.name,
            passed=passed,
            message=f"max output tokens in single call: {max_call} (limit {self.max_tokens})",
            measured=max_call,
            expected=self.max_tokens,
        )


@dataclass
class MaxTokensPerTurn:
    """Assert that a specific turn's token usage is within limit.

    Args:
        max_tokens: Maximum tokens allowed for the turn.
        turn: Turn number to check (1-indexed). If None, checks current turn.
    """

    max_tokens: int
    turn: int | None = None
    name: str = "max_tokens_per_turn"

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

        total_tokens = getattr(turn_result, "total_tokens", 0)
        passed = total_tokens <= self.max_tokens

        return AssertionResult(
            name=f"{self.name}[turn={check_turn}]",
            passed=passed,
            message=f"turn {check_turn} used {total_tokens} tokens (max {self.max_tokens})",
            measured=total_tokens,
            expected=self.max_tokens,
        )


@dataclass
class TokenBurstDetection:
    """Detect abnormal token consumption patterns (token bursts).

    A "token burst" indicates undesired behavior where token consumption
    spikes unexpectedly, often signaling:
    - Runaway agent loops
    - Excessive context accumulation
    - Prompt injection attacks causing verbose responses
    - Tool results returning unexpectedly large payloads

    This assertion fails if ANY of the following conditions are met:
    1. Any single call exceeds `absolute_max` tokens
    2. Any single call exceeds `burst_multiplier` times the average

    Args:
        absolute_max: Maximum tokens allowed in any single call (default: 50000).
        burst_multiplier: Factor above average that triggers a burst (default: 3.0).
            For example, 3.0 means a call using 3x the average tokens is a burst.
        min_calls_for_average: Minimum calls needed before burst_multiplier applies (default: 2).
            With fewer calls, only absolute_max is checked.
        mode: Which token type to check: "total" (default), "input", or "output".
            - "total": Check combined input + output tokens
            - "input": Check only input tokens (detect verbose tool results)
            - "output": Check only output tokens (detect verbose LLM responses)

    Example:
        # Fail if any call uses more than 20k tokens or 4x the average
        TokenBurstDetection(absolute_max=20000, burst_multiplier=4.0)

        # Detect when tool results are too verbose (input token burst)
        TokenBurstDetection(absolute_max=10000, mode="input")

        # Detect when LLM is too verbose (output token burst)
        TokenBurstDetection(absolute_max=5000, mode="output")
    """

    absolute_max: int = 50000
    burst_multiplier: float = 3.0
    min_calls_for_average: int = 2
    mode: str = "total"  # "total", "input", or "output"
    name: str = "token_burst_detection"

    def __call__(self, ctx: ChaosContext) -> AssertionResult:
        token_history = ctx.metrics.get_token_history()

        if not token_history:
            return AssertionResult(
                name=self.name,
                passed=True,
                message="no LLM calls to check",
            )

        # Select the token field based on mode
        if self.mode == "input":
            token_field = "input_tokens"
            mode_label = "input"
        elif self.mode == "output":
            token_field = "output_tokens"
            mode_label = "output"
        else:
            token_field = "total_tokens"
            mode_label = "total"

        max_tokens = 0
        burst_call_id = None
        burst_reason = None

        # Calculate average
        total_tokens_all = sum(h[token_field] for h in token_history)
        avg_tokens = total_tokens_all / len(token_history) if token_history else 0

        for entry in token_history:
            call_tokens = entry[token_field]

            # Check absolute limit
            if call_tokens > self.absolute_max:
                if call_tokens > max_tokens:
                    max_tokens = call_tokens
                    burst_call_id = entry["call_id"]
                    burst_reason = f"exceeded absolute max ({call_tokens} > {self.absolute_max})"

            # Check relative burst (only if enough calls for meaningful average)
            if len(token_history) >= self.min_calls_for_average and avg_tokens > 0:
                threshold = avg_tokens * self.burst_multiplier
                if call_tokens > threshold:
                    if call_tokens > max_tokens:
                        max_tokens = call_tokens
                        burst_call_id = entry["call_id"]
                        burst_reason = (
                            f"exceeded {self.burst_multiplier}x average "
                            f"({call_tokens} > {threshold:.0f}, avg={avg_tokens:.0f})"
                        )

        passed = burst_call_id is None
        name_suffix = f"[{mode_label}]" if self.mode != "total" else ""

        if passed:
            msg = (
                f"no {mode_label} token bursts detected (max={max_tokens or 0}, "
                f"avg={avg_tokens:.0f})"
            )
        else:
            msg = f"{mode_label} token burst detected: {burst_reason}"

        return AssertionResult(
            name=f"{self.name}{name_suffix}",
            passed=passed,
            message=msg,
            measured=max_tokens if not passed else (total_tokens_all // len(token_history) if token_history else 0),
            expected={"absolute_max": self.absolute_max, "burst_multiplier": self.burst_multiplier, "mode": self.mode},
        )
