"""Stream chaos types — affect LLM streaming responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.types import ChaosAction


class StreamChaos(BaseModel):
    """Base class for stream chaos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    after_chunks: int = 0
    probability: float = 1.0
    on_call: int | None = None
    after_calls: int | None = None
    provider: str | None = None
    always: bool = False
    # Turn-based triggers
    on_turn: int | None = None
    after_turns: int | None = None
    between_turns: tuple[int, int] | None = None

    _trigger: TriggerConfig = PrivateAttr()

    @model_validator(mode="after")
    def _build_trigger(self) -> StreamChaos:
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
        return ChaosPoint.STREAM

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

    def should_trigger_on_chunk(self, chunk_number: int) -> bool:
        """Check if chaos should trigger on this chunk."""
        import random

        if chunk_number >= self.after_chunks:
            if self.probability >= 1.0 or random.random() < self.probability:
                return True
        return False

    def apply(self, **kwargs: Any) -> ChaosResult:
        """Apply stream chaos. Override in subclasses."""
        return ChaosResult.proceed()


class StreamCutChaos(StreamChaos):
    """Abruptly cuts stream after N chunks."""

    def __str__(self) -> str:
        return f"stream_cut(after {self.after_chunks} chunks)"

    def apply(self, **kwargs: Any) -> ChaosResult:
        import anthropic
        import httpx

        exc = anthropic.APIConnectionError(
            message="Stream terminated unexpectedly",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        return ChaosResult.raise_exception(exc)


class StreamHangChaos(StreamChaos):
    """Hangs stream after N chunks (blocks forever)."""

    hang_seconds: float = 60.0

    def __str__(self) -> str:
        return f"stream_hang(after {self.after_chunks} chunks)"

    def apply(self, **kwargs: Any) -> ChaosResult:
        # The actual hanging is done in the stream wrapper
        # This just signals that we should hang
        return ChaosResult(action=ChaosAction.HANG)


class SlowTTFTChaos(StreamChaos):
    """Delays time-to-first-token."""

    delay: float = 0.0
    after_chunks: int = 0  # Always applies to first chunk

    def __str__(self) -> str:
        return f"slow_ttft({self.delay}s)"

    def apply(self, **kwargs: Any) -> ChaosResult:
        return ChaosResult(action=ChaosAction.DELAY, mutated=self.delay)


class SlowChunksChaos(StreamChaos):
    """Adds delay between chunks."""

    delay: float = 0.0

    def __str__(self) -> str:
        return f"slow_chunks({self.delay}s)"

    def apply(self, **kwargs: Any) -> ChaosResult:
        return ChaosResult(action=ChaosAction.DELAY, mutated=self.delay)


# Factory functions


def llm_stream_cut(after_chunks: int) -> ChaosBuilder:
    """Create a stream cut chaos."""
    return ChaosBuilder(StreamCutChaos, after_chunks=after_chunks)


def llm_stream_hang(after_chunks: int) -> ChaosBuilder:
    """Create a stream hang chaos."""
    return ChaosBuilder(StreamHangChaos, after_chunks=after_chunks)


def llm_slow_ttft(delay: float) -> ChaosBuilder:
    """Create a slow TTFT chaos."""
    return ChaosBuilder(SlowTTFTChaos, delay=delay)


def llm_slow_chunks(delay: float) -> ChaosBuilder:
    """Create a slow chunks chaos."""
    return ChaosBuilder(SlowChunksChaos, delay=delay)
