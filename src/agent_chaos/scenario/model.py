"""Scenario model (Python-first).

Scenario files are just Python modules exposing `scenario: Scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder

if TYPE_CHECKING:
    from agent_chaos.scenario.assertions import AssertionResult


@dataclass
class TurnResult:
    """Result of a completed turn, passed to dynamic input generators.

    Attributes:
        turn_number: 1-indexed turn number.
        input: The input text for this turn.
        response: The agent's response for this turn.
        success: Whether the turn completed successfully.
        duration_s: Time taken for this turn.
        llm_calls: Number of LLM calls made during this turn.
        error: Error message if turn failed.
        chaos: Turn-scoped chaos that was injected during this turn.
        assertion_results: Results of turn-scoped assertions.
        is_dynamic: Whether this turn used dynamic input.
    """

    turn_number: int
    input: str
    response: str
    success: bool
    duration_s: float
    llm_calls: int
    error: str | None = None
    chaos: list[dict[str, Any]] = field(default_factory=list)
    assertion_results: list["AssertionResult"] = field(default_factory=list)
    is_dynamic: bool = False


# Type alias for dynamic turn input generators
TurnInputGenerator = Callable[[list[TurnResult]], str]


@dataclass
class Turn:
    """A single turn in a multi-turn scenario.

    Attributes:
        input: Static string OR callable that receives conversation history.
            - String: Used directly as the turn input.
            - Callable: Receives list[TurnResult] and returns the input string.
        chaos: Turn-scoped chaos (fires only during this turn).
        assertions: Turn-scoped assertions to validate after this turn.

    Example:
        # Static input
        Turn("What's the weather in Tokyo?")

        # Dynamic input based on previous turns
        def follow_up(history: list[TurnResult]) -> str:
            if history and "sorry" in history[-1].response.lower():
                return "I want to speak to a manager!"
            return "Thanks for the help."

        Turn(input=follow_up)

        # Turn with scoped chaos and assertions
        Turn(
            input="Process my refund please.",
            chaos=[tool_error("timeout").for_tool("refund_service")],
            assertions=[TurnCompletes()],
        )
    """

    input: str | TurnInputGenerator
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    assertions: list[Any] = field(default_factory=list)

    def get_input(self, history: list[TurnResult]) -> str:
        """Resolve the input for this turn.

        Args:
            history: List of completed turn results.

        Returns:
            The input string for this turn.
        """
        if callable(self.input):
            return self.input(history)
        return self.input

    def is_dynamic(self) -> bool:
        """Check if this turn has dynamic input."""
        return callable(self.input)


@dataclass
class Scenario:
    """A single chaos scenario.

    Attributes:
        name: Unique scenario name.
        description: Human-readable description of what this scenario tests.
        agent: Callable that runs the agent under test.
            Signature: (ctx: ChaosContext, turn_input: str) -> str
        turns: List of turns defining the conversation flow.
            The agent is called once per turn with the turn input.
        chaos: Scenario-level chaos to inject (applies across all turns).
        providers: Providers to patch. Defaults to ["anthropic"].
        assertions: Scenario-level assertions to validate contracts.
        tags: Optional list of tags for UI grouping and filtering.

    Example:
        from agent_chaos import llm_rate_limit, tool_error
        from agent_chaos.scenario import Scenario, Turn, CompletesWithin

        async def my_driver(ctx: ChaosContext, turn_input: str) -> str:
            return await my_agent.run(turn_input)

        scenario = Scenario(
            name="support-flow",
            description="Tests support agent handling escalating customers",
            agent=my_driver,
            turns=[
                Turn("My order didn't arrive. I want a refund."),
                Turn(
                    input="Process it now!",
                    chaos=[tool_error("timeout").for_tool("refund")],
                ),
            ],
            chaos=[llm_rate_limit().on_turn(2).after_calls(2)],
            assertions=[CompletesWithin(timeout_s=60.0)],
        )
    """

    name: str
    description: str
    agent: Callable[..., Any]
    turns: list[Turn]
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    providers: list[str] = field(default_factory=lambda: ["anthropic"])
    assertions: list[Any] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
