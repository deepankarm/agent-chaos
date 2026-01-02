"""Context chaos types — mutate messages array."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder


# Type alias for context mutators
# The actual signature is detected at runtime in _build_trigger_and_detect_mutator
ContextMutator = Callable[..., list]


class ContextChaos(BaseModel):
    """Base class for context/messages chaos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    on_call: int | None = None
    after_calls: int | None = None
    probability: float = 1.0
    provider: str | None = None
    always: bool = False
    # Turn-based triggers
    on_turn: int | None = None
    after_turns: int | None = None
    between_turns: tuple[int, int] | None = None

    _trigger: TriggerConfig = PrivateAttr()

    @model_validator(mode="after")
    def _build_trigger(self) -> ContextChaos:
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
        return self

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


class ContextMutateChaos(ContextChaos):
    """Custom context mutation using user-provided function."""

    mutator: ContextMutator | None = None

    _accepts_ctx: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _build_trigger_and_detect_mutator(self) -> ContextMutateChaos:
        # Build trigger
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
        trigger = ""
        if self.on_call is not None:
            trigger = f" on call {self.on_call}"
        elif self.after_calls is not None:
            trigger = f" after {self.after_calls} calls"
        return f"context_mutate[{fn_name}]{trigger}"

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


def context_mutate(fn: ContextMutator) -> ChaosBuilder:
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
