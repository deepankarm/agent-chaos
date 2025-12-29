"""DeepEval integration for agent-chaos.

Wrap DeepEval metrics as agent-chaos assertions for LLM-as-judge evaluation
of agents under chaos conditions.

Usage:
    from agent_chaos.integrations.deepeval import DeepEvalAssertion, as_assertion
    from deepeval.metrics import GEval, ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCaseParams

    # Wrap any DeepEval metric
    scenario = Scenario(
        name="semantic-robustness",
        agent=my_driver,
        chaos=[tool_mutate(corrupt_weather)],
        assertions=[
            CompletesWithin(60.0),
            as_assertion(GEval(
                name="task-completion",
                criteria="Did the agent complete the user's request?",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            )),
            as_assertion(ToolCorrectnessMetric(), expected_tools=["lookup_order"]),
        ],
    )

    # Using Anthropic instead of OpenAI for evaluation:
    # DeepEval has native support - just use their AnthropicModel:
    from deepeval.models import AnthropicModel

    claude_model = AnthropicModel(model="claude-sonnet-4-20250514")
    scenario = Scenario(
        ...
        assertions=[
            as_assertion(GEval(..., model=claude_model)),
        ],
    )

Requirements:
    pip install deepeval  # Optional dependency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_chaos.scenario.assertions import AssertionResult

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext

# Lazy import check
_DEEPEVAL_AVAILABLE: bool | None = None


def _check_deepeval() -> None:
    """Check if deepeval is available, raise helpful error if not."""
    global _DEEPEVAL_AVAILABLE
    if _DEEPEVAL_AVAILABLE is None:
        try:
            import deepeval  # noqa: F401

            _DEEPEVAL_AVAILABLE = True
        except ImportError:
            _DEEPEVAL_AVAILABLE = False

    if not _DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval integration requires 'deepeval' package.\n"
            "Install it with: pip install deepeval\n"
            "Or add to dev dependencies: uv add --dev deepeval"
        )


def _extract_tools_called(ctx: "ChaosContext") -> list[Any]:
    """Extract tool calls from the conversation timeline.

    Returns a list of ToolCall objects for DeepEval.
    """
    _check_deepeval()
    from deepeval.test_case import ToolCall

    tools_called = []
    for entry in ctx.metrics.conversation:
        if entry.get("type") == "tool_call":
            tool_name = entry.get("tool_name", entry.get("name", "unknown"))
            tools_called.append(
                ToolCall(
                    name=tool_name,
                    input_parameters=entry.get("args"),
                    output=entry.get("result"),
                )
            )
        elif entry.get("type") == "tool_result":
            # Tool result comes after tool_call, we can update the last tool
            # if it exists and matches
            tool_name = entry.get("tool_name", entry.get("name"))
            if tools_called and tool_name:
                last_tool = tools_called[-1]
                if last_tool.name == tool_name and last_tool.output is None:
                    # Create new ToolCall with output since it's immutable
                    tools_called[-1] = ToolCall(
                        name=last_tool.name,
                        input_parameters=last_tool.input_parameters,
                        output=entry.get("result"),
                    )

    return tools_called


def _extract_retrieval_context(ctx: "ChaosContext") -> list[str]:
    """Extract retrieval context (tool results) from conversation."""
    context = []
    for entry in ctx.metrics.conversation:
        if entry.get("type") == "tool_result":
            result = entry.get("result")
            if result:
                context.append(str(result))
    return context


def _format_conversation_for_eval(ctx: "ChaosContext") -> tuple[str, str]:
    """Format multi-turn conversation for LLM evaluation.

    Returns:
        Tuple of (input_text, output_text) representing the full conversation.
        - input_text: All user messages with turn context
        - output_text: All assistant responses with turn context
    """
    if not ctx.turn_results or len(ctx.turn_results) <= 1:
        # Single turn - use simple input/output
        return ctx.agent_input or "", ctx.agent_output or ""

    # Multi-turn: format as conversation
    input_parts = []
    output_parts = []

    for tr in ctx.turn_results:
        turn_label = f"[Turn {tr.turn_number}]"
        input_parts.append(f"{turn_label} User: {tr.input}")
        if tr.response:
            output_parts.append(f"{turn_label} Assistant: {tr.response}")

    return "\n".join(input_parts), "\n".join(output_parts)


def build_llm_test_case(
    ctx: "ChaosContext",
    expected_tools: list[str] | None = None,
    expected_output: str | None = None,
    turn: int | None = None,
) -> Any:
    """Build a DeepEval LLMTestCase from ChaosContext.

    Args:
        ctx: The chaos context with agent run data
        expected_tools: Optional list of tool names that should be called
        expected_output: Optional expected output for comparison
        turn: If specified, build test case for a specific turn only.
              If None and multi-turn, includes full conversation context.

    Returns:
        deepeval.test_case.LLMTestCase
    """
    _check_deepeval()
    from deepeval.test_case import LLMTestCase, ToolCall

    # Get input and output
    if turn is not None:
        # Specific turn requested
        turn_result = ctx.get_turn_result(turn)
        if turn_result is None:
            raise ValueError(f"Turn {turn} not found in context")
        input_text = turn_result.input
        actual_output = turn_result.response
    else:
        # No specific turn - use full conversation context for multi-turn
        input_text, actual_output = _format_conversation_for_eval(ctx)

    # Extract tools called
    tools_called = _extract_tools_called(ctx)

    # Build expected_tools as ToolCall objects
    expected_tools_objs = None
    if expected_tools:
        expected_tools_objs = [ToolCall(name=name) for name in expected_tools]

    # Extract retrieval context
    retrieval_context = _extract_retrieval_context(ctx)

    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        tools_called=tools_called if tools_called else None,
        expected_tools=expected_tools_objs,
        retrieval_context=retrieval_context if retrieval_context else None,
    )


def build_conversational_test_case(
    ctx: "ChaosContext",
    scenario: str | None = None,
    expected_outcome: str | None = None,
    chatbot_role: str | None = None,
) -> Any:
    """Build a DeepEval ConversationalTestCase from ChaosContext.

    Args:
        ctx: The chaos context with multi-turn conversation data
        scenario: Description of the test scenario
        expected_outcome: What should happen in this scenario
        chatbot_role: Role description for the chatbot

    Returns:
        deepeval.test_case.ConversationalTestCase
    """
    _check_deepeval()
    from deepeval.test_case import ConversationalTestCase
    from deepeval.test_case import Turn as DeepEvalTurn

    turns = []

    # Group conversation entries by turn
    current_turn_entries: list[dict[str, Any]] = []
    current_turn_number = 0

    for entry in ctx.metrics.conversation:
        entry_type = entry.get("type")
        turn_num = entry.get("turn_number", 0)

        if entry_type == "turn_start":
            # New turn starting
            if current_turn_entries:
                # Process previous turn
                _add_turns_from_entries(turns, current_turn_entries)
            current_turn_entries = []
            current_turn_number = turn_num
        elif entry_type == "turn_end":
            # Turn ending, process entries
            _add_turns_from_entries(turns, current_turn_entries)
            current_turn_entries = []
        elif entry_type in ("user", "assistant", "tool_call", "tool_result"):
            current_turn_entries.append(entry)

    # Process any remaining entries
    if current_turn_entries:
        _add_turns_from_entries(turns, current_turn_entries)

    # If no structured turns, fall back to simple input/output
    if not turns and ctx.agent_input and ctx.agent_output:
        from deepeval.test_case import Turn as DeepEvalTurn

        turns = [
            DeepEvalTurn(role="user", content=ctx.agent_input),
            DeepEvalTurn(role="assistant", content=ctx.agent_output),
        ]

    return ConversationalTestCase(
        turns=turns,
        scenario=scenario,
        expected_outcome=expected_outcome,
        chatbot_role=chatbot_role,
    )


def _add_turns_from_entries(turns: list[Any], entries: list[dict[str, Any]]) -> None:
    """Convert conversation entries to DeepEval Turn objects."""
    _check_deepeval()
    from deepeval.test_case import ToolCall
    from deepeval.test_case import Turn as DeepEvalTurn

    user_content = None
    assistant_content = None
    tools_called = []

    for entry in entries:
        entry_type = entry.get("type")

        if entry_type == "user":
            user_content = entry.get("content", "")
        elif entry_type == "assistant":
            assistant_content = entry.get("content", "")
        elif entry_type == "tool_call":
            tool_name = entry.get("tool_name", entry.get("name", "unknown"))
            tools_called.append(
                ToolCall(
                    name=tool_name,
                    input_parameters=entry.get("args"),
                )
            )
        elif entry_type == "tool_result":
            # Update last tool with result
            tool_name = entry.get("tool_name", entry.get("name"))
            if tools_called and tool_name:
                for i, tc in enumerate(tools_called):
                    if tc.name == tool_name:
                        tools_called[i] = ToolCall(
                            name=tc.name,
                            input_parameters=tc.input_parameters,
                            output=entry.get("result"),
                        )
                        break

    # Add user turn
    if user_content:
        turns.append(DeepEvalTurn(role="user", content=user_content))

    # Add assistant turn with tools
    if assistant_content or tools_called:
        turns.append(
            DeepEvalTurn(
                role="assistant",
                content=assistant_content or "",
                tools_called=tools_called if tools_called else None,
            )
        )


@dataclass
class DeepEvalAssertion:
    """Wrap a DeepEval metric as an agent-chaos assertion.

    Works with LLMTestCase-based metrics like GEval, ToolCorrectnessMetric, etc.

    For per-turn evaluation, add assertions to Turn.assertions.
    For whole-conversation evaluation, add assertions to Scenario.assertions.

    Args:
        metric: Any DeepEval metric with a measure() method
        threshold: Override the metric's threshold (optional)
        expected_tools: List of tool names expected to be called (for ToolCorrectnessMetric)
        expected_output: Expected output string (for comparison metrics)
        turn: Evaluate a specific turn only (for multi-turn scenarios)
        name: Custom name for the assertion (defaults to metric class name)

    Example:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        # Per-turn: add to Turn.assertions
        Turn(
            input="What's my order status?",
            assertions=[as_assertion(GEval(...))],  # Evaluates just this turn
        )

        # Whole conversation: add to Scenario.assertions
        Scenario(
            ...,
            assertions=[as_assertion(GEval(...))],  # Evaluates all turns together
        )
    """

    metric: Any  # BaseMetric from DeepEval or Callable that returns one
    threshold: float | None = None
    expected_tools: list[str] | None = None
    expected_output: str | None = None
    turn: int | None = None
    name: str | None = None
    _resolved_metric: Any = field(default=None, repr=False, compare=False)

    def _get_metric(self) -> Any:
        """Get the metric, resolving it lazily if it's a callable."""
        if self._resolved_metric is not None:
            return self._resolved_metric

        # Check if metric is a callable (factory function)
        if callable(self.metric) and not hasattr(self.metric, "measure"):
            self._resolved_metric = self.metric()
        else:
            self._resolved_metric = self.metric
        return self._resolved_metric

    def _evaluate_single(
        self, ctx: "ChaosContext", turn: int | None = None
    ) -> tuple[float, str, bool]:
        """Evaluate metric for a single test case.

        Returns:
            Tuple of (score, reason, passed)
        """
        metric = self._get_metric()

        # Build test case
        test_case = build_llm_test_case(
            ctx,
            expected_tools=self.expected_tools,
            expected_output=self.expected_output,
            turn=turn,
        )

        # Override threshold if specified
        threshold = self.threshold or getattr(metric, "threshold", 0.5)
        if self.threshold is not None and hasattr(metric, "threshold"):
            metric.threshold = self.threshold

        # Run the metric
        metric.measure(test_case)

        score = getattr(metric, "score", 0.0)
        reason = getattr(metric, "reason", "")
        passed = score >= threshold

        return score, reason, passed

    def __call__(
        self, ctx: "ChaosContext", turn_number: int | None = None
    ) -> AssertionResult:
        """Evaluate the DeepEval metric against the chaos context.

        Args:
            ctx: The chaos context with agent run data.
            turn_number: If provided (by runner for Turn.assertions), evaluate
                just that turn. Otherwise uses self.turn or full conversation.
        """
        _check_deepeval()

        threshold = self.threshold or getattr(self._get_metric(), "threshold", 0.5)

        # Determine which turn to evaluate:
        # 1. turn_number passed by runner (for Turn.assertions)
        # 2. self.turn set at construction time
        # 3. None = full conversation
        eval_turn = turn_number if turn_number is not None else self.turn

        try:
            score, reason, passed = self._evaluate_single(ctx, turn=eval_turn)
            return AssertionResult(
                name=self._get_name(),
                passed=passed,
                message=reason or f"score={score:.2f} (threshold={threshold:.2f})",
                measured=score,
                expected=threshold,
            )
        except Exception as e:
            return AssertionResult(
                name=self._get_name(),
                passed=False,
                message=f"DeepEval metric failed: {e}",
                measured=None,
                expected=threshold,
            )

    def _get_name(self) -> str:
        """Get the assertion name."""
        if self.name:
            return self.name
        # Try to get name without resolving (for lazy metrics)
        if self._resolved_metric is not None:
            metric_name = getattr(self._resolved_metric, "name", None)
            if metric_name:
                return f"deepeval:{metric_name}"
            return f"deepeval:{self._resolved_metric.__class__.__name__}"
        # Fallback for callable metrics before resolution
        if callable(self.metric) and not hasattr(self.metric, "measure"):
            return f"deepeval:{self.metric.__name__}"
        metric_name = getattr(self.metric, "name", None)
        if metric_name:
            return f"deepeval:{metric_name}"
        return f"deepeval:{self.metric.__class__.__name__}"


@dataclass
class ConversationalDeepEvalAssertion:
    """Wrap a DeepEval conversational metric as an agent-chaos assertion.

    Works with ConversationalTestCase-based metrics like ConversationalGEval,
    RoleAdherenceMetric, etc.

    Args:
        metric: A DeepEval conversational metric
        threshold: Override the metric's threshold (optional)
        scenario: Description of the test scenario
        expected_outcome: What should happen in this scenario
        chatbot_role: Role description for the chatbot
        name: Custom name for the assertion

    Example:
        from deepeval.metrics import ConversationalGEval
        from deepeval.test_case import ConversationalTestCaseParams

        ConversationalDeepEvalAssertion(
            metric=ConversationalGEval(
                name="conversation-quality",
                criteria="Is the conversation coherent and helpful?",
                evaluation_params=[ConversationalTestCaseParams.TURNS],
            ),
            scenario="Customer asking for a refund",
            expected_outcome="Agent processes refund or explains why not",
        )
    """

    metric: Any  # Conversational metric from DeepEval or Callable that returns one
    threshold: float | None = None
    scenario: str | None = None
    expected_outcome: str | None = None
    chatbot_role: str | None = None
    name: str | None = None
    _resolved_metric: Any = field(default=None, repr=False, compare=False)

    def _get_metric(self) -> Any:
        """Get the metric, resolving it lazily if it's a callable."""
        if self._resolved_metric is not None:
            return self._resolved_metric

        # Check if metric is a callable (factory function)
        if callable(self.metric) and not hasattr(self.metric, "measure"):
            self._resolved_metric = self.metric()
        else:
            self._resolved_metric = self.metric
        return self._resolved_metric

    def __call__(self, ctx: "ChaosContext") -> AssertionResult:
        """Evaluate the conversational metric against the chaos context."""
        _check_deepeval()

        # Resolve the metric lazily
        metric = self._get_metric()

        # Build conversational test case
        test_case = build_conversational_test_case(
            ctx,
            scenario=self.scenario,
            expected_outcome=self.expected_outcome,
            chatbot_role=self.chatbot_role,
        )

        # Override threshold if specified
        if self.threshold is not None and hasattr(metric, "threshold"):
            metric.threshold = self.threshold

        # Run the metric
        try:
            metric.measure(test_case)
        except Exception as e:
            return AssertionResult(
                name=self._get_name(),
                passed=False,
                message=f"DeepEval conversational metric failed: {e}",
                measured=None,
                expected=self.threshold,
            )

        # Get results
        score = getattr(metric, "score", 0.0)
        reason = getattr(metric, "reason", "")
        threshold = self.threshold or getattr(metric, "threshold", 0.5)
        passed = score >= threshold

        return AssertionResult(
            name=self._get_name(),
            passed=passed,
            message=reason or f"score={score:.2f} (threshold={threshold:.2f})",
            measured=score,
            expected=threshold,
        )

    def _get_name(self) -> str:
        """Get the assertion name."""
        if self.name:
            return self.name
        # Try to get name without resolving (for lazy metrics)
        if self._resolved_metric is not None:
            metric_name = getattr(self._resolved_metric, "name", None)
            if metric_name:
                return f"deepeval:{metric_name}"
            return f"deepeval:{self._resolved_metric.__class__.__name__}"
        # Fallback for callable metrics before resolution
        if callable(self.metric) and not hasattr(self.metric, "measure"):
            return f"deepeval:{self.metric.__name__}"
        metric_name = getattr(self.metric, "name", None)
        if metric_name:
            return f"deepeval:{metric_name}"
        return f"deepeval:{self.metric.__class__.__name__}"


def as_assertion(
    metric: Any,
    threshold: float | None = None,
    expected_tools: list[str] | None = None,
    expected_output: str | None = None,
    turn: int | None = None,
    name: str | None = None,
    # Conversational params
    scenario: str | None = None,
    expected_outcome: str | None = None,
    chatbot_role: str | None = None,
) -> DeepEvalAssertion | ConversationalDeepEvalAssertion:
    """Convenience function to wrap any DeepEval metric as an assertion.

    Automatically detects whether the metric is conversational or single-turn
    based on the metric class.

    For per-turn evaluation, add to Turn.assertions.
    For whole-conversation evaluation, add to Scenario.assertions.

    Args:
        metric: Any DeepEval metric
        threshold: Override the metric's threshold
        expected_tools: Expected tool names (for ToolCorrectnessMetric)
        expected_output: Expected output text
        turn: Specific turn to evaluate
        name: Custom assertion name
        scenario: Scenario description (conversational)
        expected_outcome: Expected outcome (conversational)
        chatbot_role: Chatbot role (conversational)

    Example:
        from deepeval.metrics import GEval
        from agent_chaos.integrations.deepeval import as_assertion

        # Per-turn: evaluates just that turn
        Turn(input="...", assertions=[as_assertion(GEval(...))])

        # Whole conversation: evaluates all turns together
        Scenario(..., assertions=[as_assertion(GEval(...))])
    """
    _check_deepeval()

    # Try to detect if it's a conversational metric
    metric_class_name = metric.__class__.__name__.lower()
    is_conversational = (
        "conversational" in metric_class_name
        or "role" in metric_class_name
        or scenario is not None
        or expected_outcome is not None
        or chatbot_role is not None
    )

    if is_conversational:
        return ConversationalDeepEvalAssertion(
            metric=metric,
            threshold=threshold,
            scenario=scenario,
            expected_outcome=expected_outcome,
            chatbot_role=chatbot_role,
            name=name,
        )

    return DeepEvalAssertion(
        metric=metric,
        threshold=threshold,
        expected_tools=expected_tools,
        expected_output=expected_output,
        turn=turn,
        name=name,
    )


# Convenience aliases
wrap_metric = as_assertion
