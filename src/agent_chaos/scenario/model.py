"""Scenario model (Python-first).

Two scenario types enforce the baseline → chaos variant pattern:
- BaselineScenario: Pure journey definition, NO chaos allowed
- ChaosScenario: Chaos variant, created via BaselineScenario.variant()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder

if TYPE_CHECKING:
    from agent_chaos.scenario.assertions import AssertionResult


@dataclass
class at:
    """Target chaos/assertions to a specific turn in a variant.

    Used with BaselineScenario.variant() to inject chaos or assertions
    at specific turns.

    Attributes:
        turn: Turn index (0-based).
        chaos: Chaos events to inject at this turn.
        assertions: Assertions to check after this turn.

    Example:
        baseline.variant(
            name="refund-fails",
            turns=[
                at(2, chaos=[tool_error().for_tool("refund")]),
                at(4, assertions=[turn_coherence]),
            ],
        )
    """

    turn: int
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    assertions: list[Any] = field(default_factory=list)


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
        input_tokens: Total input tokens consumed during this turn.
        output_tokens: Total output tokens generated during this turn.
        total_tokens: Total tokens (input + output) for this turn.
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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# Type alias for dynamic turn input generators
TurnInputGenerator = Callable[[list[TurnResult]], str]


@dataclass
class Turn:
    """A single turn in a multi-turn scenario.

    Attributes:
        input: Static string OR callable that receives conversation history.
            - String: Used directly as the turn input.
            - Callable: Receives list[TurnResult] and returns the input string.
        chaos: Turn-scoped chaos (only valid in ChaosScenario).
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
    """

    input: str | TurnInputGenerator
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    assertions: list[Any] = field(default_factory=list)

    def get_input(self, history: list[TurnResult]) -> str:
        """Resolve the input for this turn."""
        if callable(self.input):
            return self.input(history)
        return self.input

    def is_dynamic(self) -> bool:
        """Check if this turn has dynamic input."""
        return callable(self.input)


@dataclass
class BaselineScenario:
    """A baseline scenario - defines the journey WITHOUT chaos.

    BaselineScenario represents the "happy path" - what the agent should do
    under normal conditions. Use .variant() to create ChaosScenario instances
    that test resilience under failure conditions.

    Attributes:
        name: Unique scenario name.
        description: Human-readable description.
        agent: Callable that runs the agent under test.
            Signature: (ctx: ChaosContext, turn_input: str) -> str
        turns: List of turns defining the conversation flow.
            Turns in a BaselineScenario cannot have chaos.
        providers: Providers to patch. Defaults to ["anthropic"].
        assertions: Scenario-level assertions to validate contracts.
        tags: Optional list of tags for UI grouping and filtering.
        meta: Optional metadata dictionary.

    Example:
        from agent_chaos import BaselineScenario, Turn

        customer_journey = BaselineScenario(
            name="customer-journey",
            description="Standard customer support flow",
            agent=run_agent,
            turns=[
                Turn("Check my order status"),
                Turn("I want a refund"),
                Turn("Process it please"),
            ],
            assertions=[CompletesSuccessfully()],
        )

        # Create chaos variants
        llm_fails = customer_journey.variant(
            name="customer-journey-llm-fails",
            chaos=[llm_error().after_calls(2)],
        )
    """

    name: str
    description: str
    agent: Callable[..., Any]
    turns: list[Turn]
    providers: list[str] = field(default_factory=lambda: ["anthropic"])
    assertions: list[Any] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # NO chaos field - enforced by design

    def __post_init__(self) -> None:
        """Validate that no turns have chaos."""
        for i, turn in enumerate(self.turns):
            if turn.chaos:
                raise ValueError(
                    f"BaselineScenario '{self.name}' cannot have chaos in turns. "
                    f"Turn {i} has chaos defined. Use .variant() to add chaos."
                )

    def variant(
        self,
        *,
        name: str,
        description: str | None = None,
        chaos: list[Chaos | ChaosBuilder] | None = None,
        assertions: list[Any] | None = None,
        turns: list[at] | None = None,
        tags: list[str] | None = None,
    ) -> "ChaosScenario":
        """Create a chaos variant of this baseline.

        This method creates a ChaosScenario based on this baseline, allowing
        you to test the same agent journey under different failure conditions.

        Args:
            name: Name for the chaos variant.
            description: Optional description (defaults to parent's).
            chaos: Scenario-level chaos to inject.
            assertions: Additional assertions (appended to parent's).
            turns: List of `at()` objects specifying turn-level chaos/assertions.
            tags: Additional tags (appended to parent's).

        Returns:
            A new ChaosScenario with chaos applied.

        Example:
            # Turn-level chaos
            tool_fails = baseline.variant(
                name="tool-timeout",
                turns=[at(2, chaos=[tool_error().for_tool("refund")])],
            )

            # Global chaos
            llm_fails = baseline.variant(
                name="llm-rate-limited",
                chaos=[llm_rate_limit().after_calls(3)],
            )

            # Both
            combined = baseline.variant(
                name="chaos-storm",
                chaos=[llm_error()],
                turns=[at(1, chaos=[tool_timeout()])],
                assertions=[ErrorHandledGracefully()],
            )
        """
        # Build turn modifications lookup: {turn_index: at(...)}
        turn_mods: dict[int, at] = {}
        for mod in turns or []:
            turn_mods[mod.turn] = mod

        # Create turns with chaos applied from at() specs
        new_turns = []
        for i, turn in enumerate(self.turns):
            mod = turn_mods.get(i)
            if mod:
                # Add chaos and assertions from at() spec
                new_turns.append(
                    Turn(
                        input=turn.input,
                        chaos=list(mod.chaos),
                        assertions=list(turn.assertions) + list(mod.assertions),
                    )
                )
            else:
                # Copy turn without chaos
                new_turns.append(
                    Turn(
                        input=turn.input,
                        chaos=[],
                        assertions=list(turn.assertions),
                    )
                )

        return ChaosScenario(
            name=name,
            description=description or self.description,
            agent=self.agent,
            turns=new_turns,
            chaos=chaos or [],
            providers=list(self.providers),
            assertions=list(self.assertions) + (assertions or []),
            tags=list(self.tags) + (tags or []),
            parent=self.name,
        )


@dataclass
class ChaosScenario:
    """A chaos scenario - tests resilience under failure conditions.

    ChaosScenario is typically created via BaselineScenario.variant(),
    though it can be constructed directly for advanced use cases.

    Attributes:
        name: Unique scenario name.
        description: Human-readable description.
        agent: Callable that runs the agent under test.
        turns: List of turns (can have chaos).
        chaos: Scenario-level chaos to inject.
        providers: Providers to patch.
        assertions: Scenario-level assertions.
        tags: Optional list of tags.
        parent: Name of the parent BaselineScenario.

    Note:
        ChaosScenario does not have a variant() method.
        Chaos variants should derive from BaselineScenario only,
        keeping the hierarchy flat: Baseline → Variant(s).
    """

    name: str
    description: str
    agent: Callable[..., Any]
    turns: list[Turn]
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    providers: list[str] = field(default_factory=lambda: ["anthropic"])
    assertions: list[Any] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    parent: str = ""

    # NO variant() method - flat hierarchy only


# Type alias for runner compatibility
Scenario = BaselineScenario | ChaosScenario
