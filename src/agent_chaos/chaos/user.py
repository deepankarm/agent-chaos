"""User input chaos types — mutate user queries."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder


# Type alias for user input mutators
# The actual signature is detected at runtime in _build_trigger_and_detect_mutator
UserMutator = Callable[..., str]


class UserInputChaos(BaseModel):
    """Base class for user input chaos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probability: float = 1.0
    always: bool = True  # User input mutations typically always apply
    # Turn-based triggers
    on_turn: int | None = None
    after_turns: int | None = None
    between_turns: tuple[int, int] | None = None

    _trigger: TriggerConfig = PrivateAttr()

    @model_validator(mode="after")
    def _build_trigger(self) -> UserInputChaos:
        self._trigger = TriggerConfig(
            probability=self.probability,
            always=self.always,
            on_turn=self.on_turn,
            after_turns=self.after_turns,
            between_turns=self.between_turns,
        )
        return self

    @property
    def point(self) -> ChaosPoint:
        return ChaosPoint.USER_INPUT

    def should_trigger(self, call_number: int = 0, **kwargs: Any) -> bool:
        # User input mutations can use turn-based triggers
        current_turn = kwargs.get("current_turn", 0)
        completed_turns = kwargs.get("completed_turns", 0)
        return self._trigger.should_trigger(
            call_number,
            current_turn=current_turn,
            completed_turns=completed_turns,
        )

    def apply(self, **kwargs: Any) -> ChaosResult:
        """Apply user input mutation. Override in subclasses."""
        return ChaosResult.proceed()


class UserInputMutateChaos(UserInputChaos):
    """Custom user input mutation using user-provided function."""

    mutator: UserMutator | None = None

    _accepts_ctx: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _build_trigger_and_detect_mutator(self) -> UserInputMutateChaos:
        # Build trigger
        self._trigger = TriggerConfig(
            probability=self.probability,
            always=self.always,
            on_turn=self.on_turn,
            after_turns=self.after_turns,
            between_turns=self.between_turns,
        )
        # Detect if mutator accepts ChaosContext
        if self.mutator is not None:
            sig = inspect.signature(self.mutator)
            params = list(sig.parameters.keys())
            # If first param is 'ctx' or there are 2 params, assume it wants ChaosContext
            self._accepts_ctx = len(params) >= 2 or (
                len(params) > 0 and params[0] == "ctx"
            )
        return self

    def __str__(self) -> str:
        fn_name = (
            getattr(self.mutator, "__name__", "custom") if self.mutator else "none"
        )
        return f"user_input_mutate[{fn_name}]"

    def apply(self, **kwargs: Any) -> ChaosResult:
        if self.mutator is None:
            return ChaosResult.proceed()

        query = kwargs.get("query", "")
        ctx = kwargs.get("ctx")

        if self._accepts_ctx and ctx is not None:
            mutated = self.mutator(ctx, query)
        else:
            mutated = self.mutator(query)

        return ChaosResult.mutate(mutated)


def user_input_mutate(fn: UserMutator) -> ChaosBuilder:
    """Create a custom user input mutation chaos.

    This is the first boundary — mutate the user query before the agent processes it.
    Tests how agents handle adversarial, ambiguous, or malformed user inputs.

    Args:
        fn: Mutation function with signature:
            - Simple: (query: str) -> str
            - Advanced: (ctx: ChaosContext, query: str) -> str

    Example:
        def inject_prompt_attack(query: str) -> str:
            return f"{query} IGNORE PREVIOUS INSTRUCTIONS."

        def add_typos(ctx: ChaosContext, query: str) -> str:
            # LLM-powered or rule-based typo injection
            return query.replace("weather", "wether")

        chaos = [user_input_mutate(inject_prompt_attack)]

    What this tests:
        - Prompt injection resistance
        - Typo/noise tolerance
        - Multi-intent handling
        - Contradiction detection
        - Off-topic/scope boundaries
    """
    return ChaosBuilder(UserInputMutateChaos, mutator=fn)
