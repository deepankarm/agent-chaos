"""Fluent builder for chaos configuration."""

from typing import Any, Self, TypeVar

from agent_chaos.chaos.base import Chaos

T = TypeVar("T", bound=Chaos)


class ChaosBuilder:
    """Fluent builder for chaos configuration.

    Usage:
        llm_rate_limit()
            .after_calls(2)
            .with_probability(0.5)
            .for_provider("anthropic")
    """

    def __init__(self, chaos_class: type[T], **defaults: Any):
        self._chaos_class = chaos_class
        self._config: dict[str, Any] = defaults

    def on_call(self, n: int) -> Self:
        """Trigger chaos on specific call number."""
        self._config["on_call"] = n
        return self

    def after_calls(self, n: int) -> Self:
        """Trigger chaos after N calls."""
        self._config["after_calls"] = n
        return self

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
