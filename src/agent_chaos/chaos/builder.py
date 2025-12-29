"""Fluent builder for chaos configuration."""

from typing import Any, Self, TypeVar

from agent_chaos.chaos.base import Chaos

T = TypeVar("T", bound=Chaos)


class ChaosBuilder:
    """Fluent builder for chaos configuration.

    Supports both call-based and turn-based triggers:

    Call-based (within a turn):
        llm_rate_limit().on_call(2)        # On 2nd LLM call
        llm_rate_limit().after_calls(2)    # After 2nd LLM call

    Turn-based (across turns):
        llm_rate_limit().on_turn(2)        # On turn 2
        llm_rate_limit().after_turns(1)    # After turn 1 completes
        history_mutate(fn).between_turns(1, 2)  # Between turns 1 and 2

    Combined:
        llm_rate_limit().on_turn(2).after_calls(1)
        # On turn 2, after the 1st LLM call in that turn
    """

    def __init__(self, chaos_class: type[T], **defaults: Any):
        self._chaos_class = chaos_class
        self._config: dict[str, Any] = defaults

    # --- Call-based triggers (within a turn) ---

    def on_call(self, n: int) -> Self:
        """Trigger chaos on specific call number (within current turn)."""
        self._config["on_call"] = n
        return self

    def after_calls(self, n: int) -> Self:
        """Trigger chaos after N calls (within current turn)."""
        self._config["after_calls"] = n
        return self

    # --- Turn-based triggers ---

    def on_turn(self, n: int) -> Self:
        """Trigger chaos on specific turn number (1-indexed).

        Can be combined with call-based triggers:
            llm_rate_limit().on_turn(2).after_calls(1)
            # Triggers on turn 2, after the 1st LLM call in that turn
        """
        self._config["on_turn"] = n
        return self

    def after_turns(self, n: int) -> Self:
        """Trigger chaos after N turns complete.

        Example:
            llm_rate_limit().after_turns(1)
            # Triggers after turn 1 completes (i.e., starting from turn 2)
        """
        self._config["after_turns"] = n
        return self

    def between_turns(self, after_turn: int, before_turn: int) -> Self:
        """Trigger chaos between two turns (for history/context mutations).

        This is primarily for between-turn chaos like history_mutate()
        that operates on the conversation history between turns.

        Example:
            history_mutate(inject_fake).between_turns(1, 2)
            # Mutates history after turn 1 completes, before turn 2 starts
        """
        self._config["between_turns"] = (after_turn, before_turn)
        return self

    # --- Other triggers ---

    def with_probability(self, p: float) -> Self:
        """Trigger chaos with given probability (0.0-1.0)."""
        self._config["probability"] = p
        return self

    def for_provider(self, provider: str) -> Self:
        """Target specific provider."""
        self._config["provider"] = provider
        return self

    def for_tool(self, tool_name: str) -> Self:
        """Target specific tool (for tool chaos)."""
        self._config["tool_name"] = tool_name
        return self

    def always(self) -> Self:
        """Trigger chaos on every call."""
        self._config["always"] = True
        return self

    def build(self) -> T:
        """Build the chaos instance."""
        return self._chaos_class(**self._config)

    # Allow using builder directly without .build()
    # by implementing the Chaos protocol methods as pass-through

    @property
    def point(self):
        """Delegate to built chaos."""
        return self.build().point

    def should_trigger(self, call_number: int, **kwargs: Any) -> bool:
        """Delegate to built chaos."""
        return self.build().should_trigger(call_number, **kwargs)

    def apply(self, **kwargs: Any):
        """Delegate to built chaos."""
        return self.build().apply(**kwargs)

    # Store reference to built instance for efficiency
    _built: T | None = None

    def _get_or_build(self) -> T:
        if self._built is None:
            self._built = self.build()
        return self._built
