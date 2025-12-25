"""Chaos injection logic — routes chaos to the right injection points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_chaos.chaos.base import Chaos, ChaosPoint, ChaosResult
from agent_chaos.chaos.builder import ChaosBuilder

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext


def _build_if_needed(chaos_or_builder: Chaos | ChaosBuilder) -> Chaos:
    """Convert ChaosBuilder to Chaos if needed."""
    if isinstance(chaos_or_builder, ChaosBuilder):
        return chaos_or_builder.build()
    return chaos_or_builder


class ChaosInjector:
    """Routes chaos to the right injection points."""

    def __init__(self, chaos: list[Chaos | ChaosBuilder] | None = None):
        all_chaos = [_build_if_needed(c) for c in (chaos or [])]

        self._llm_chaos: list[Chaos] = []
        self._stream_chaos: list[Chaos] = []
        self._tool_chaos: list[Chaos] = []
        self._context_chaos: list[Chaos] = []

        for c in all_chaos:
            point = c.point
            if point == ChaosPoint.LLM_CALL:
                self._llm_chaos.append(c)
            elif point == ChaosPoint.STREAM:
                self._stream_chaos.append(c)
            elif point == ChaosPoint.TOOL_RESULT:
                self._tool_chaos.append(c)
            elif point == ChaosPoint.MESSAGES:
                self._context_chaos.append(c)

        self._call_count = 0
        self._chaos_used: set[tuple[str, int]] = set()
        self._ctx: ChaosContext | None = None

    def set_context(self, ctx: ChaosContext) -> None:
        """Set the ChaosContext reference for advanced chaos functions."""
        self._ctx = ctx

    def increment_call(self) -> int:
        """Increment and return call count."""
        self._call_count += 1
        return self._call_count

    def next_llm_chaos(self, provider: str) -> ChaosResult | None:
        """Get the next LLM chaos to apply, if any."""
        call_number = self._call_count

        for idx, chaos in enumerate(self._llm_chaos):
            chaos_id = ("llm", idx)
            if chaos_id in self._chaos_used:
                continue

            if chaos.should_trigger(call_number, provider=provider):
                self._chaos_used.add(chaos_id)
                return chaos.apply(provider=provider)

        return None

    def get_stream_chaos(self) -> list[Chaos]:
        """Get all stream chaos objects."""
        return self._stream_chaos

    def ttft_delay(self) -> float | None:
        """Get TTFT delay if configured."""
        from agent_chaos.chaos.stream import SlowTTFTChaos

        for chaos in self._stream_chaos:
            if isinstance(chaos, SlowTTFTChaos):
                return chaos.delay
        return None

    def should_hang(self, chunk_count: int) -> bool:
        """Check if stream should hang at this chunk."""
        from agent_chaos.chaos.stream import StreamHangChaos

        for chaos in self._stream_chaos:
            if isinstance(chaos, StreamHangChaos):
                if chaos.should_trigger_on_chunk(chunk_count):
                    return True
        return False

    def should_cut(self, chunk_count: int) -> bool:
        """Check if stream should be cut at this chunk."""
        from agent_chaos.chaos.stream import StreamCutChaos

        for chaos in self._stream_chaos:
            if isinstance(chaos, StreamCutChaos):
                if chaos.should_trigger_on_chunk(chunk_count):
                    return True
        return False

    def chunk_delay(self) -> float | None:
        """Get delay between chunks if configured."""
        from agent_chaos.chaos.stream import SlowChunksChaos

        for chaos in self._stream_chaos:
            if isinstance(chaos, SlowChunksChaos):
                return chaos.delay
        return None

    def should_corrupt(self, chunk_count: int) -> bool:
        """Check if chunk should be corrupted."""
        # Not implemented in new chaos yet
        return False

    def corruption_type(self) -> str:
        """Get corruption type if corruption is configured."""
        return "truncate_text"

    # --- Tool Chaos ---

    def next_tool_chaos(
        self, tool_name: str, result: str
    ) -> tuple[ChaosResult, Chaos] | None:
        """Get the next tool chaos to apply, if any. Returns (result, chaos_obj)."""
        call_number = self._call_count

        for chaos in self._tool_chaos:
            if chaos.should_trigger(call_number, tool_name=tool_name):
                return (
                    chaos.apply(tool_name=tool_name, result=result, ctx=self._ctx),
                    chaos,
                )

        return None

    def should_mutate_tools(self) -> bool:
        """Check if tool mutations should be applied."""
        return bool(self._tool_chaos)

    def get_tool_mutation(self, tool_name: str) -> None:
        """Get mutation for a specific tool (legacy interface, returns None)."""
        return None

    def next_context_chaos(self, messages: list) -> tuple[ChaosResult, Chaos] | None:
        """Get the next context chaos to apply, if any. Returns (result, chaos_obj)."""
        call_number = self._call_count

        for chaos in self._context_chaos:
            if chaos.should_trigger(call_number):
                return (chaos.apply(messages=messages, ctx=self._ctx), chaos)

        return None
