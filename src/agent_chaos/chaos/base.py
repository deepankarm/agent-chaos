"""Base chaos types and protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from agent_chaos.types import ChaosAction, ChaosPoint

# Re-export for backwards compatibility
__all__ = ["ChaosPoint", "ChaosAction", "ChaosResult", "Chaos", "TriggerConfig"]


class ChaosResult(BaseModel):
    """Result of applying chaos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: ChaosAction
    exception: Exception | None = None
    mutated: Any | None = None

    @classmethod
    def proceed(cls) -> ChaosResult:
        """Continue without chaos."""
        return cls(action=ChaosAction.PROCEED)

    @classmethod
    def raise_exception(cls, exc: Exception) -> ChaosResult:
        """Raise an exception."""
        return cls(action=ChaosAction.RAISE, exception=exc)

    @classmethod
    def mutate(cls, value: Any) -> ChaosResult:
        """Return mutated value."""
        return cls(action=ChaosAction.MUTATE, mutated=value)


@runtime_checkable
class Chaos(Protocol):
    """Protocol for all chaos types.

    Each Chaos knows:
    - Its injection point (where it applies)
    - When to trigger (call number, probability)
    - What to do (raise exception, mutate, etc.)
    """

    @property
    def point(self) -> ChaosPoint:
        """Where this chaos applies."""
        ...

    def should_trigger(self, call_number: int, **kwargs: Any) -> bool:
        """Check if chaos should trigger on this call."""
        ...

    def apply(self, **kwargs: Any) -> ChaosResult:
        """Apply the chaos and return result."""
        ...


class TriggerConfig(BaseModel):
    """Common triggering configuration.

    Supports both call-based and turn-based triggers:
    - Call-based: on_call, after_calls (operate within current turn)
    - Turn-based: on_turn, after_turns, between_turns (operate across turns)

    Turn triggers and call triggers can be combined:
        llm_rate_limit().on_turn(2).after_calls(1)
        # Triggers on turn 2, after the first LLM call within that turn
    """

    # Call-based triggers (within a turn)
    on_call: int | None = None
    after_calls: int | None = None

    # Turn-based triggers
    on_turn: int | None = None  # Fire on specific turn (1-indexed)
    after_turns: int | None = None  # Fire after N turns complete
    between_turns: tuple[int, int] | None = None  # Fire between turn a and turn b

    # Other triggers
    probability: float | None = None
    provider: str | None = None
    always: bool = False

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, v: float | None) -> float | None:
        """Validate probability is between 0 and 1."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("probability must be between 0.0 and 1.0")
        return v

    @field_validator("on_call", "after_calls", "on_turn", "after_turns")
    @classmethod
    def validate_positive(cls, v: int | None) -> int | None:
        """Validate that call/turn numbers are positive."""
        if v is not None and v < 1:
            raise ValueError("call and turn numbers must be >= 1")
        return v

    def should_trigger(
        self,
        call_number: int,
        provider: str | None = None,
        current_turn: int = 0,
        completed_turns: int = 0,
        **kwargs: Any,
    ) -> bool:
        """Check if chaos should trigger.

        Args:
            call_number: Current LLM call number (1-indexed).
            provider: Provider name (e.g., "anthropic").
            current_turn: Current turn number (1-indexed, 0 = no turn).
            completed_turns: Number of completed turns.
            **kwargs: Additional context (tool_name, etc.)
        """
        import random

        # Provider filter
        if self.provider is not None and provider != self.provider:
            return False

        # Always trigger
        if self.always:
            return True

        # --- Turn-based triggers ---

        # On specific turn
        if self.on_turn is not None:
            if current_turn != self.on_turn:
                return False
            # If only on_turn is set (no call triggers), trigger immediately
            if self.on_call is None and self.after_calls is None:
                # Trigger once per turn (use probability or first call)
                if self.probability is not None:
                    return random.random() < self.probability
                return True  # Trigger on first opportunity in this turn

        # After N turns complete
        if self.after_turns is not None:
            if completed_turns < self.after_turns:
                return False
            # If only after_turns is set (no call triggers), trigger immediately
            if self.on_call is None and self.after_calls is None and self.on_turn is None:
                if self.probability is not None:
                    return random.random() < self.probability
                return True

        # Between turns (for history mutations)
        if self.between_turns is not None:
            after_turn, before_turn = self.between_turns
            # This should trigger between the completion of after_turn
            # and the start of before_turn
            if not (completed_turns >= after_turn and current_turn == 0):
                return False
            # Trigger once
            if self.probability is not None:
                return random.random() < self.probability
            return True

        # --- Call-based triggers (within current turn) ---

        # On specific call
        if self.on_call is not None:
            if call_number != self.on_call:
                return False
            # If on_turn is also set, we already checked it above
            return True

        # After N calls
        if self.after_calls is not None:
            if call_number <= self.after_calls:
                return False
            # If on_turn is also set, we already checked it above
            return True

        # Probability-based (standalone)
        if self.probability is not None:
            return random.random() < self.probability

        # Default: don't trigger
        return False
