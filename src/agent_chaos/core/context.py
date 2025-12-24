"""Chaos context manager and context object."""

from contextlib import contextmanager
from typing import Any, Iterator

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore


class ChaosContext:
    """Context object providing access to injector and metrics."""

    def __init__(
        self,
        name: str,
        injector: ChaosInjector,
        metrics: MetricsStore,
        session_id: str,
    ):
        self.name = name
        self.injector = injector
        self.metrics = metrics
        self.session_id = session_id
        self.result: Any | None = None
        self.error: str | None = None
        self.elapsed_s: float | None = None


@contextmanager
def chaos_context(
    name: str,
    chaos: list[Chaos | ChaosBuilder] | None = None,
    providers: list[str] | None = None,
    emit_events: bool = False,
    event_sink: Any | None = None,
) -> Iterator[ChaosContext]:
    """Context manager for scoped chaos injection.

    Introduce a little chaos at every boundary of your agent.

    Args:
        name: Name for this chaos context (shown in UI)
        chaos: List of chaos to inject
        providers: List of providers to patch (default: ["anthropic"])
        emit_events: If True, emit events to the UI dashboard
        event_sink: Optional event sink for artifact persistence (e.g. JSONL)

    Yields:
        ChaosContext with injector and metrics access

    Example:
        from agent_chaos import (
            chaos_context,
            llm_rate_limit,
            llm_stream_cut,
            tool_error,
        )

        with chaos_context(
            name="test",
            chaos=[
                llm_rate_limit().after_calls(2),
                llm_stream_cut(after_chunks=10),
                tool_error("down").for_tool("weather"),
            ],
        ) as ctx:
            result = my_agent.run("...")
    """
    from agent_chaos.patch.patcher import ChaosPatcher

    injector = ChaosInjector(chaos=chaos)
    metrics = MetricsStore()

    session_id = ""
    if emit_events:
        from agent_chaos.ui.events import event_bus

        metrics.set_event_bus(event_bus)
        session_id = event_bus.start_session(name)
        metrics.set_trace_context(event_bus.trace_id, name)

    if event_sink is not None:
        metrics.set_event_sink(event_sink)
        if hasattr(event_sink, "start_trace") and callable(
            getattr(event_sink, "start_trace")
        ):
            trace_ctx = event_sink.start_trace(name)
            metrics.set_trace_context(trace_ctx.trace_id, trace_ctx.trace_name)
            session_id = trace_ctx.trace_id

    patcher = ChaosPatcher(injector, metrics)
    providers = providers or ["anthropic"]

    ctx = ChaosContext(
        name=name, injector=injector, metrics=metrics, session_id=session_id
    )
    injector.set_context(ctx)

    try:
        if "anthropic" in providers:
            patcher._patch_anthropic()
        if "openai" in providers:
            patcher._patch_openai()
        if "gemini" in providers:
            patcher._patch_gemini()
        yield ctx
    finally:
        patcher.unpatch_all()
        if emit_events:
            from agent_chaos.ui.events import event_bus

            event_bus.end_session()
        if event_sink is not None and hasattr(event_sink, "end_trace"):
            try:
                event_sink.end_trace(
                    metrics._trace_id,
                    metrics._trace_name,
                    {
                        "total_calls": metrics.total_calls,
                        "failed_calls": sum(
                            1
                            for c in metrics.call_history
                            if not c.get("success", True)
                        ),
                        "chaos_count": len(metrics.faults_injected),
                    },
                )
            except Exception:
                pass
        if event_sink is not None and hasattr(event_sink, "close"):
            try:
                event_sink.close()
            except Exception:
                pass
