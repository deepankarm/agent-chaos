"""Pydantic Evals integration for agent-chaos.

Wrap pydantic-evals evaluators (especially LLMJudge) as agent-chaos assertions
for LLM-as-judge evaluation of agents under chaos conditions.

Usage:
    from agent_chaos.integrations.pydantic_evals import as_assertion
    from pydantic_evals.evaluators import LLMJudge

    scenario = ChaosScenario(
        name="semantic-robustness",
        agent=my_agent,
        chaos=[tool_error("get_weather")],
        assertions=[
            CompletesWithin(60.0),
            as_assertion(
                LLMJudge(
                    rubric="The agent gracefully handles errors and informs the user",
                    model="anthropic:claude-sonnet-4-5",
                    include_input=True,
                ),
                threshold=0.7,
            ),
        ],
    )

Requirements:
    pip install pydantic-evals  # Optional dependency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_chaos.scenario.assertions import AssertionResult

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext

_PYDANTIC_EVALS_AVAILABLE: bool | None = None


def _check_pydantic_evals() -> None:
    """Check if pydantic-evals is available, raise helpful error if not."""
    global _PYDANTIC_EVALS_AVAILABLE
    if _PYDANTIC_EVALS_AVAILABLE is None:
        try:
            import pydantic_evals  # noqa: F401

            _PYDANTIC_EVALS_AVAILABLE = True
        except ImportError:
            _PYDANTIC_EVALS_AVAILABLE = False

    if not _PYDANTIC_EVALS_AVAILABLE:
        raise ImportError(
            "Pydantic Evals integration requires 'pydantic-evals' package.\n"
            "Install it with: pip install pydantic-evals\n"
            "Or add to dev dependencies: uv add --dev pydantic-evals"
        )


def _format_conversation_for_eval(ctx: "ChaosContext") -> tuple[str, str]:
    """Format multi-turn conversation for LLM evaluation.

    Returns:
        Tuple of (input_text, output_text) representing the full conversation.
    """
    if not ctx.turn_results or len(ctx.turn_results) <= 1:
        return ctx.agent_input or "", ctx.agent_output or ""

    input_parts = []
    output_parts = []

    for tr in ctx.turn_results:
        turn_label = f"[Turn {tr.turn_number}]"
        input_parts.append(f"{turn_label} User: {tr.input}")
        if tr.response:
            output_parts.append(f"{turn_label} Assistant: {tr.response}")

    return "\n".join(input_parts), "\n".join(output_parts)


def _extract_chaos_context(ctx: "ChaosContext", turn: int | None = None) -> str | None:
    """Extract chaos injection info for evaluator context."""
    chaos_events = []
    for entry in ctx.metrics.conv.entries:
        if entry.get("type") == "chaos":
            event_turn = entry.get("turn_number", 0)
            if turn is not None and event_turn > turn:
                continue
            fault_type = entry.get("fault_type", "unknown")
            target = entry.get("target_tool", "")
            chaos_point = entry.get("chaos_point", "")
            if target:
                chaos_events.append(f"- {fault_type} on {target} (turn {event_turn})")
            else:
                chaos_events.append(
                    f"- {fault_type} at {chaos_point} (turn {event_turn})"
                )

    if not chaos_events:
        return None

    return "Chaos/errors injected during this scenario:\n" + "\n".join(chaos_events)


def build_evaluator_context(
    ctx: "ChaosContext",
    expected_output: str | None = None,
    turn: int | None = None,
    include_chaos_info: bool = True,
) -> Any:
    """Build a pydantic-evals EvaluatorContext from ChaosContext.

    Args:
        ctx: The chaos context with agent run data.
        expected_output: Optional expected output for comparison.
        turn: If specified, build context for a specific turn only.
        include_chaos_info: If True, include chaos injection info in metadata.

    Returns:
        pydantic_evals.evaluators.context.EvaluatorContext
    """
    _check_pydantic_evals()
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.otel._errors import SpanTreeRecordingError

    if turn is not None:
        turn_result = ctx.get_turn_result(turn)
        if turn_result is None:
            raise ValueError(f"Turn {turn} not found in context")
        input_text = turn_result.input
        output_text = turn_result.response
        duration = turn_result.duration_s
    else:
        input_text, output_text = _format_conversation_for_eval(ctx)
        duration = ctx.elapsed_s or 0.0

    metadata: dict[str, Any] = {}
    if include_chaos_info:
        chaos_info = _extract_chaos_context(ctx, turn)
        if chaos_info:
            metadata["chaos_context"] = chaos_info
        else:
            metadata["chaos_context"] = "No errors were injected."

    return EvaluatorContext(
        name=ctx.name,
        inputs=input_text,
        metadata=metadata if metadata else None,
        expected_output=expected_output,
        output=output_text,
        duration=duration,
        _span_tree=SpanTreeRecordingError(
            "Span recording not available in agent-chaos"
        ),
        attributes={},
        metrics={},
    )


def _parse_evaluator_output(
    output: Any,
    threshold: float | None,
    evaluator_name: str,
) -> tuple[bool, float | None, str]:
    """Parse pydantic-evals evaluator output into (passed, score, message).

    Args:
        output: The raw output from evaluator.evaluate_sync().
        threshold: Optional threshold for score-based passing.
        evaluator_name: Name of the evaluator for message formatting.

    Returns:
        Tuple of (passed, score, message).
    """
    _check_pydantic_evals()
    from pydantic_evals.evaluators.evaluator import EvaluationReason

    if isinstance(output, bool):
        return output, None, f"{evaluator_name}: {'passed' if output else 'failed'}"

    if isinstance(output, (int, float)):
        score = float(output)
        if threshold is not None:
            passed = score >= threshold
            return passed, score, f"score={score:.2f} (threshold={threshold:.2f})"
        return score >= 0.5, score, f"score={score:.2f}"

    if isinstance(output, str):
        return True, None, output

    if isinstance(output, EvaluationReason):
        value = output.value
        reason = output.reason or ""
        if isinstance(value, bool):
            return (
                value,
                None,
                reason or f"{evaluator_name}: {'passed' if value else 'failed'}",
            )
        if isinstance(value, (int, float)):
            score = float(value)
            if threshold is not None:
                passed = score >= threshold
                msg = reason or f"score={score:.2f} (threshold={threshold:.2f})"
                return passed, score, msg
            return score >= 0.5, score, reason or f"score={score:.2f}"
        return True, None, reason or str(value)

    if isinstance(output, dict):
        all_passed = True
        scores = []
        messages = []
        for key, val in output.items():
            sub_passed, sub_score, sub_msg = _parse_evaluator_output(
                val, threshold, key
            )
            all_passed = all_passed and sub_passed
            if sub_score is not None:
                scores.append(sub_score)
            messages.append(f"{key}: {sub_msg}")

        avg_score = sum(scores) / len(scores) if scores else None
        return all_passed, avg_score, "; ".join(messages)

    return True, None, str(output)


@dataclass
class PydanticEvalsAssertion:
    """Wrap a pydantic-evals Evaluator as an agent-chaos assertion.

    Args:
        evaluator: Any pydantic-evals Evaluator (e.g., LLMJudge, IsInstance).
        threshold: Score threshold for pass/fail (for score-based evaluators).
        expected_output: Optional expected output for comparison evaluators.
        turn: If specified, evaluate only this turn (1-indexed).
        name: Custom name for the assertion.
        include_chaos_info: Include chaos injection info in evaluator context.

    Example:
        from pydantic_evals.evaluators import LLMJudge
        from agent_chaos.integrations.pydantic_evals import PydanticEvalsAssertion

        assertion = PydanticEvalsAssertion(
            evaluator=LLMJudge(
                rubric="Response is accurate and helpful",
                model="anthropic:claude-sonnet-4-5",
                include_input=True,
            ),
            threshold=0.7,
        )
    """

    evaluator: Any
    threshold: float | None = None
    expected_output: str | None = None
    turn: int | None = None
    name: str | None = None
    include_chaos_info: bool = True

    def __call__(
        self, ctx: "ChaosContext", turn_number: int | None = None
    ) -> AssertionResult:
        """Evaluate the pydantic-evals evaluator against the chaos context.

        Args:
            ctx: The chaos context with agent run data.
            turn_number: If provided (by runner for Turn.assertions), evaluate
                just that turn. Otherwise uses self.turn or full conversation.
        """
        _check_pydantic_evals()

        eval_turn = turn_number if turn_number is not None else self.turn

        try:
            eval_ctx = build_evaluator_context(
                ctx,
                expected_output=self.expected_output,
                turn=eval_turn,
                include_chaos_info=self.include_chaos_info,
            )

            output = self.evaluator.evaluate_sync(eval_ctx)
            passed, score, message = _parse_evaluator_output(
                output, self.threshold, self._get_name()
            )

            return AssertionResult(
                name=self._get_name(),
                passed=passed,
                message=message,
                measured=score,
                expected=self.threshold,
            )

        except Exception as e:
            return AssertionResult(
                name=self._get_name(),
                passed=False,
                message=f"Pydantic Evals evaluator failed: {e}",
                measured=None,
                expected=self.threshold,
            )

    def _get_name(self) -> str:
        """Get the assertion name."""
        if self.name:
            return self.name
        # Try get_serialization_name() first (pydantic-evals preferred method)
        if hasattr(self.evaluator, "get_serialization_name"):
            try:
                return f"pydantic-evals:{self.evaluator.get_serialization_name()}"
            except Exception:
                pass
        # Fall back to class name
        class_name = self.evaluator.__class__.__name__
        return f"pydantic-evals:{class_name}"


def as_assertion(
    evaluator: Any,
    threshold: float | None = None,
    expected_output: str | None = None,
    turn: int | None = None,
    name: str | None = None,
    include_chaos_info: bool = True,
) -> PydanticEvalsAssertion:
    """Wrap any pydantic-evals Evaluator as an agent-chaos assertion.

    Args:
        evaluator: Any pydantic-evals Evaluator (e.g., LLMJudge).
        threshold: Score threshold for pass/fail (default: 0.5 for scores).
        expected_output: Expected output for comparison evaluators.
        turn: Specific turn to evaluate (1-indexed).
        name: Custom assertion name.
        include_chaos_info: Include chaos injection info in evaluator context.

    Returns:
        PydanticEvalsAssertion that can be used in scenario assertions.

    Example:
        from pydantic_evals.evaluators import LLMJudge
        from agent_chaos.integrations.pydantic_evals import as_assertion

        scenario = ChaosScenario(
            name="test",
            agent=my_agent,
            assertions=[
                as_assertion(
                    LLMJudge(
                        rubric="Agent handles errors gracefully",
                        model="anthropic:claude-sonnet-4-5",
                        include_input=True,
                    ),
                    threshold=0.7,
                ),
            ],
        )
    """
    _check_pydantic_evals()

    return PydanticEvalsAssertion(
        evaluator=evaluator,
        threshold=threshold,
        expected_output=expected_output,
        turn=turn,
        name=name,
        include_chaos_info=include_chaos_info,
    )
