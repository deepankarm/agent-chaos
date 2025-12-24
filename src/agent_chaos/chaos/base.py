"""Base chaos types and protocols."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ChaosPoint(str, Enum):
    """Injection points for chaos."""

    LLM_CALL = "llm_call"  # Before LLM call → raise exception
    STREAM = "stream"  # During streaming → hang/cut/slow
    TOOL_RESULT = "tool_result"  # After tool returns → mutate result
    MESSAGES = "messages"  # Before LLM call → mutate messages array


@dataclass
class ChaosResult:
    """Result of applying chaos."""

    action: str  # "proceed", "raise", "mutate"
    exception: Exception | None = None
    mutated: Any | None = None

    @classmethod
    def proceed(cls) -> "ChaosResult":
        """Continue without chaos."""
        return cls(action="proceed")

    @classmethod
    def raise_exception(cls, exc: Exception) -> "ChaosResult":
        """Raise an exception."""
        return cls(action="raise", exception=exc)

    @classmethod
    def mutate(cls, value: Any) -> "ChaosResult":
        """Return mutated value."""
        return cls(action="mutate", mutated=value)


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


@dataclass
class TriggerConfig:
    """Common triggering configuration."""

    on_call: int | None = None
    after_calls: int | None = None
    probability: float | None = None
    provider: str | None = None
    always: bool = False

    def should_trigger(self, call_number: int, provider: str | None = None) -> bool:
        """Check if chaos should trigger."""
        import random

        # Provider filter
        if self.provider is not None and provider != self.provider:
            return False

        # Always trigger
        if self.always:
            return True

        # On specific call
        if self.on_call is not None:
            return call_number == self.on_call

        # After N calls
        if self.after_calls is not None:
            return call_number > self.after_calls

        # Probability-based
        if self.probability is not None:
            return random.random() < self.probability

        # Default: don't trigger
        return False
