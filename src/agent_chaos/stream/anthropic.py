"""Anthropic stream wrapper for fault injection."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import httpx

from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore


def _stream_connection_error() -> anthropic.APIConnectionError:
    """Create an APIConnectionError for stream cut."""
    return anthropic.APIConnectionError(
        message="Stream terminated unexpectedly",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


class BaseStreamFaultMixin(ABC):
    """Mixin providing common stream chaos injection logic."""

    _injector: ChaosInjector
    _metrics: MetricsStore
    _call_id: str
    _chunk_count: int

    def _init_fault_state(self):
        """Initialize fault tracking state."""
        self._chunk_count = 0
        self._slow_chunks_recorded = False

    def _check_ttft_delay_sync(self):
        """Check and apply TTFT delay (sync)."""
        if self._chunk_count == 1:
            # Check if delay will be applied (for chaos tracking)
            delay = self._injector.ttft_delay()

            # Get call start time from metrics
            call_info = self._metrics._active_calls.get(self._call_id)
            if call_info:
                call_start_time = call_info["start_time"]
                ttft = time.monotonic() - call_start_time
                self._metrics.record_ttft(ttft, self._call_id, is_delayed=bool(delay))

            # Apply delay fault if configured
            if delay:
                time.sleep(delay)

    async def _check_ttft_delay_async(self):
        """Check and apply TTFT delay (async)."""
        if self._chunk_count == 1:
            # Check if delay will be applied (for chaos tracking)
            delay = self._injector.ttft_delay()

            # Get call start time from metrics
            call_info = self._metrics._active_calls.get(self._call_id)
            if call_info:
                call_start_time = call_info["start_time"]
                ttft = time.monotonic() - call_start_time
                self._metrics.record_ttft(ttft, self._call_id, is_delayed=bool(delay))

            # Apply delay fault if configured
            if delay:
                await asyncio.sleep(delay)

    def _check_stream_hang_sync(self):
        """Check and apply stream hang (sync)."""
        if self._injector.should_hang(self._chunk_count):
            self._metrics.record_hang(self._chunk_count, self._call_id)
            while True:
                time.sleep(1)

    async def _check_stream_hang_async(self):
        """Check and apply stream hang (async)."""
        if self._injector.should_hang(self._chunk_count):
            self._metrics.record_hang(self._chunk_count, self._call_id)
            while True:
                await asyncio.sleep(1)

    def _check_stream_cut(self):
        """Check and raise stream cut error."""
        if self._injector.should_cut(self._chunk_count):
            self._metrics.record_stream_cut(self._chunk_count, self._call_id)
            raise _stream_connection_error()

    def _check_slow_chunks_sync(self):
        """Check and apply slow chunk delay (sync)."""
        if delay := self._injector.chunk_delay():
            # Record chaos event once
            if not self._slow_chunks_recorded:
                self._slow_chunks_recorded = True
                self._metrics.record_slow_chunks(delay * 1000, self._call_id)
            time.sleep(delay)

    async def _check_slow_chunks_async(self):
        """Check and apply slow chunk delay (async)."""
        if delay := self._injector.chunk_delay():
            # Record chaos event once
            if not self._slow_chunks_recorded:
                self._slow_chunks_recorded = True
                self._metrics.record_slow_chunks(delay * 1000, self._call_id)
            await asyncio.sleep(delay)

    def _check_corruption(self, event):
        """Check and apply event corruption."""
        if self._injector.should_corrupt(self._chunk_count):
            event = self._corrupt_event(event)
            self._metrics.record_corruption(self._chunk_count)
        return event

    def _corrupt_event(self, event):
        """Corrupt event based on configured type."""
        match self._injector.corruption_type():
            case "wrong_event_type":
                if hasattr(event, "type"):
                    event.type = "error"
            case "empty_delta":
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    event.delta.text = ""
            case "truncate_text":
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    event.delta.text = event.delta.text[:5]
        return event

    def _record_chunk(self):
        """Record chunk in metrics."""
        self._metrics.record_chunk(self._chunk_count)

    def _end_call_success(self):
        """Mark call as successful."""
        # Record final stream stats before ending the call.
        try:
            self._metrics.record_stream_stats(
                self._call_id, chunk_count=self._chunk_count, provider="anthropic"
            )
        except Exception:
            pass
        self._metrics.end_call(self._call_id, success=True)


class ChaosAnthropicStream:
    """Wraps Anthropic's sync MessageStreamManager."""

    def __init__(
        self,
        inner: Any,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id

    def __enter__(self) -> "ChaosMessageStream":
        stream = self._inner.__enter__()
        return ChaosMessageStream(stream, self._injector, self._metrics, self._call_id)

    def __exit__(self, *args):
        return self._inner.__exit__(*args)


class ChaosMessageStream(BaseStreamFaultMixin):
    """Wraps sync stream iterator with fault injection."""

    def __init__(
        self,
        inner: Any,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id
        self._init_fault_state()

    def __iter__(self):
        return self

    def __next__(self):
        self._chunk_count += 1
        self._check_ttft_delay_sync()
        self._check_stream_hang_sync()
        self._check_stream_cut()
        self._check_slow_chunks_sync()

        try:
            event = next(self._inner)
        except StopIteration:
            self._end_call_success()
            raise

        event = self._check_corruption(event)
        self._record_chunk()
        return event

    @property
    def text_stream(self):
        """Wrap text_stream with fault injection."""
        return ChaosTextStream(
            self._inner.text_stream, self._injector, self._metrics, self._call_id
        )

    def get_final_message(self):
        return self._inner.get_final_message()

    def get_final_text(self):
        return self._inner.get_final_text()


class ChaosTextStream(BaseStreamFaultMixin):
    """Wraps sync text_stream iterator."""

    def __init__(
        self,
        inner,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = iter(inner)
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id
        self._init_fault_state()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        self._chunk_count += 1
        self._check_ttft_delay_sync()
        self._check_stream_hang_sync()
        self._check_stream_cut()
        self._check_slow_chunks_sync()

        try:
            text = next(self._inner)
        except StopIteration:
            self._end_call_success()
            raise

        self._record_chunk()
        return text


class ChaosAsyncAnthropicStream:
    """Wraps Anthropic's async MessageStreamManager."""

    def __init__(
        self,
        inner: Any,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id

    async def __aenter__(self) -> "ChaosAsyncMessageStream":
        stream = await self._inner.__aenter__()
        return ChaosAsyncMessageStream(
            stream, self._injector, self._metrics, self._call_id
        )

    async def __aexit__(self, *args):
        return await self._inner.__aexit__(*args)


class ChaosAsyncMessageStream(BaseStreamFaultMixin):
    """Wraps async stream iterator with fault injection."""

    def __init__(
        self,
        inner: Any,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id
        self._init_fault_state()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._chunk_count += 1
        await self._check_ttft_delay_async()
        await self._check_stream_hang_async()
        self._check_stream_cut()
        await self._check_slow_chunks_async()

        try:
            event = await self._inner.__anext__()
        except StopAsyncIteration:
            self._end_call_success()
            raise

        event = self._check_corruption(event)
        self._record_chunk()
        return event

    @property
    def text_stream(self):
        return ChaosAsyncTextStream(
            self._inner.text_stream, self._injector, self._metrics, self._call_id
        )

    async def get_final_message(self):
        return await self._inner.get_final_message()

    async def get_final_text(self):
        return await self._inner.get_final_text()


class ChaosAsyncStreamResponse(BaseStreamFaultMixin):
    """Wraps AsyncStream response (from create with stream=True).

    Used when calling client.beta.messages.create(..., stream=True).
    """

    def __init__(
        self,
        inner: Any,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id
        self._init_fault_state()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._chunk_count += 1
        await self._check_ttft_delay_async()
        await self._check_stream_hang_async()
        self._check_stream_cut()
        await self._check_slow_chunks_async()

        try:
            event = await self._inner.__anext__()
        except StopAsyncIteration:
            self._end_call_success()
            raise

        event = self._check_corruption(event)
        self._record_chunk()
        return event

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._inner.__aexit__(*args)

    async def close(self):
        if hasattr(self._inner, "close"):
            await self._inner.close()

    @property
    def response(self):
        return self._inner.response


class ChaosAsyncTextStream(BaseStreamFaultMixin):
    """Wraps async text_stream iterator."""

    def __init__(
        self,
        inner,
        injector: ChaosInjector,
        metrics: MetricsStore,
        call_id: str,
    ):
        self._inner = inner
        self._injector = injector
        self._metrics = metrics
        self._call_id = call_id
        self._init_fault_state()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        self._chunk_count += 1
        await self._check_ttft_delay_async()
        await self._check_stream_hang_async()
        self._check_stream_cut()
        await self._check_slow_chunks_async()

        try:
            text = await self._inner.__anext__()
        except StopAsyncIteration:
            self._end_call_success()
            raise

        self._record_chunk()
        return text
