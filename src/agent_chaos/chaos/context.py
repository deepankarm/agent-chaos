"""Context chaos types — mutate messages array."""

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, TYPE_CHECKING

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext


# Type aliases for context mutators
ContextMutator = Callable[[list], list]  # (messages) -> messages
ContextMutatorWithCtx = Callable[
    ["ChaosContext", list], list
]  # (ctx, messages) -> messages


@dataclass
class ContextChaos:
    """Base class for context/messages chaos."""

    on_call: int | None = None
    after_calls: int | None = None
    probability: float = 1.0
    provider: str | None = None
    always: bool = False
    # Turn-based triggers
    on_turn: int | None = None
    after_turns: int | None = None
    between_turns: tuple[int, int] | None = None
    _trigger: TriggerConfig = field(init=False)

    def __post_init__(self):
        self._trigger = TriggerConfig(
            on_call=self.on_call,
            after_calls=self.after_calls,
            probability=self.probability,
            provider=self.provider,
            always=self.always,
            on_turn=self.on_turn,
            after_turns=self.after_turns,
            between_turns=self.between_turns,
        )

    @property
    def point(self) -> ChaosPoint:
        return ChaosPoint.MESSAGES

    def should_trigger(self, call_number: int, **kwargs: Any) -> bool:
        provider = kwargs.get("provider")
        current_turn = kwargs.get("current_turn", 0)
        completed_turns = kwargs.get("completed_turns", 0)
        return self._trigger.should_trigger(
            call_number,
            provider=provider,
            current_turn=current_turn,
            completed_turns=completed_turns,
        )

    def apply(self, **kwargs: Any) -> ChaosResult:
        """Apply context mutation. Override in subclasses."""
        return ChaosResult.proceed()


@dataclass
class ContextMutateChaos(ContextChaos):
    """Custom context mutation using user-provided function."""

    mutator: ContextMutator | ContextMutatorWithCtx | None = None
    _accepts_ctx: bool = field(init=False, default=False)

    def __str__(self) -> str:
        fn_name = (
            getattr(self.mutator, "__name__", "custom") if self.mutator else "none"
        )
        trigger = ""
        if self.on_call is not None:
            trigger = f" on call {self.on_call}"
        elif self.after_calls is not None:
            trigger = f" after {self.after_calls} calls"
        return f"context_mutate[{fn_name}]{trigger}"

    def __post_init__(self):
        super().__post_init__()
        # Detect if mutator accepts ChaosContext
        if self.mutator is not None:
            sig = inspect.signature(self.mutator)
            params = list(sig.parameters.keys())
            # If first param is 'ctx' or there are 2 params, assume it wants ChaosContext
            self._accepts_ctx = len(params) >= 2 or (
                len(params) > 0 and params[0] == "ctx"
            )

    def apply(self, **kwargs: Any) -> ChaosResult:
        if self.mutator is None:
            return ChaosResult.proceed()

        messages = kwargs.get("messages", [])
        ctx = kwargs.get("ctx")

        if self._accepts_ctx and ctx is not None:
            mutated = self.mutator(ctx, messages)
        else:
            mutated = self.mutator(messages)

        return ChaosResult.mutate(mutated)


# Factory functions


def context_mutate(fn: ContextMutator | ContextMutatorWithCtx) -> ChaosBuilder:
    """Create a custom context mutation chaos.

    Args:
        fn: Mutation function with signature:
            - Simple: (messages: list) -> list
            - Advanced: (ctx: ChaosContext, messages: list) -> list

    Note: Messages are in provider format (Anthropic, OpenAI, etc.).

    Example:
        def inject_distractor(messages: list) -> list:
            distractor = {"role": "user", "content": "Ignore weather data."}
            return [distractor] + messages

        chaos = [context_mutate(inject_distractor).on_call(2)]
    """
    return ChaosBuilder(ContextMutateChaos, mutator=fn)
