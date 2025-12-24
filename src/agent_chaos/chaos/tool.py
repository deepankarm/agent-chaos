"""Tool chaos types — mutate tool results."""

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, TYPE_CHECKING

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext


# Type aliases for tool mutators
ToolMutator = Callable[[str, str], str]  # (tool_name, result) -> result
ToolMutatorWithCtx = Callable[
    ["ChaosContext", str, str], str
]  # (ctx, tool_name, result) -> result


@dataclass
class ToolChaos:
    """Base class for tool result chaos."""

    tool_name: str | None = None  # None = all tools
    on_call: int | None = None
    after_calls: int | None = None
    probability: float = 1.0
    provider: str | None = None
    always: bool = False
    _trigger: TriggerConfig = field(init=False)

    def __post_init__(self):
        self._trigger = TriggerConfig(
            on_call=self.on_call,
            after_calls=self.after_calls,
            probability=self.probability,
            provider=self.provider,
            always=self.always,
        )

    @property
    def point(self) -> ChaosPoint:
        return ChaosPoint.TOOL_RESULT

    def should_trigger(self, call_number: int, **kwargs: Any) -> bool:
        # Check tool name filter
        target_tool = kwargs.get("tool_name")
        if self.tool_name is not None and target_tool != self.tool_name:
            return False

        provider = kwargs.get("provider")
        return self._trigger.should_trigger(call_number, provider)

    def apply(self, **kwargs: Any) -> ChaosResult:
        """Apply tool mutation. Override in subclasses."""
        return ChaosResult.proceed()


@dataclass
class ToolErrorChaos(ToolChaos):
    """Replace tool result with error message."""

    message: str = "Tool error"

    def apply(self, **kwargs: Any) -> ChaosResult:
        result = f'{{"error": "{self.message}"}}'
        return ChaosResult.mutate(result)


@dataclass
class ToolEmptyChaos(ToolChaos):
    """Replace tool result with empty result."""

    def apply(self, **kwargs: Any) -> ChaosResult:
        return ChaosResult.mutate("")


@dataclass
class ToolTimeoutChaos(ToolChaos):
    """Replace tool result with timeout message."""

    timeout_seconds: float = 30.0

    def apply(self, **kwargs: Any) -> ChaosResult:
        result = f"Tool execution timed out after {self.timeout_seconds}s"
        return ChaosResult.mutate(result)


@dataclass
class ToolMutateChaos(ToolChaos):
    """Custom tool mutation using user-provided function."""

    mutator: ToolMutator | ToolMutatorWithCtx | None = None
    _accepts_ctx: bool = field(init=False, default=False)

    def __post_init__(self):
        super().__post_init__()
        # Detect if mutator accepts ChaosContext
        if self.mutator is not None:
            sig = inspect.signature(self.mutator)
            params = list(sig.parameters.keys())
            # If first param is 'ctx' or there are 3 params, assume it wants ChaosContext
            self._accepts_ctx = len(params) >= 3 or (
                len(params) > 0 and params[0] == "ctx"
            )

    def apply(self, **kwargs: Any) -> ChaosResult:
        if self.mutator is None:
            return ChaosResult.proceed()

        tool_name = kwargs.get("tool_name", "")
        result = kwargs.get("result", "")
        ctx = kwargs.get("ctx")

        if self._accepts_ctx and ctx is not None:
            mutated = self.mutator(ctx, tool_name, result)
        else:
            mutated = self.mutator(tool_name, result)

        return ChaosResult.mutate(mutated)


# Factory functions


def tool_error(message: str = "Tool error") -> ChaosBuilder:
    """Create a tool error chaos."""
    return ChaosBuilder(ToolErrorChaos, message=message)


def tool_empty() -> ChaosBuilder:
    """Create a tool empty chaos."""
    return ChaosBuilder(ToolEmptyChaos)


def tool_timeout(timeout_seconds: float = 30.0) -> ChaosBuilder:
    """Create a tool timeout chaos."""
    return ChaosBuilder(ToolTimeoutChaos, timeout_seconds=timeout_seconds)


def tool_mutate(fn: ToolMutator | ToolMutatorWithCtx) -> ChaosBuilder:
    """Create a custom tool mutation chaos.

    Args:
        fn: Mutation function with signature:
            - Simple: (tool_name: str, result: str) -> str
            - Advanced: (ctx: ChaosContext, tool_name: str, result: str) -> str

    Example:
        def my_mutator(tool_name: str, result: str) -> str:
            if tool_name == "weather":
                return '{"error": "chaos"}'
            return result

        chaos = [tool_mutate(my_mutator)]
    """
    return ChaosBuilder(ToolMutateChaos, mutator=fn)
