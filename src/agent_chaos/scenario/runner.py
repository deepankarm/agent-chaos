"""Scenario runner (CLI-first)."""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from pathlib import Path
from typing import Any, Callable

from agent_chaos.core.context import chaos_context, ChaosContext
from agent_chaos.event.jsonl import JsonlEventSink
from agent_chaos.scenario.assertions import AssertionResult
from agent_chaos.scenario.model import Scenario
from agent_chaos.scenario.report import RunReport


def _run_agent(agent: Callable[..., Any], ctx: ChaosContext) -> Any:
    """Invoke scenario driver.

    We prefer calling `driver(ctx)` because most drivers need access to the ChaosContext.
    If the driver is truly no-arg, fall back to `driver()` only when that succeeds.
    """
    if "ctx" in inspect.signature(agent).parameters:
        return agent(ctx)
    return agent()


def _run_maybe_await(value: Any) -> Any:
    """Run awaitables for async drivers (CLI-friendly).

    This keeps Scenario drivers flexible: they can be sync or async.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def run_scenario(
    scenario: Scenario,
    *,
    artifacts_dir: str | Path | None = None,
    seed: int | None = None,
    record_events: bool = True,
) -> RunReport:
    """Run a scenario and return a RunReport.

    If artifacts_dir is provided, writes:
    - events.jsonl (if record_events)
    - scorecard.json (always)
    """
    if seed is not None:
        random.seed(seed)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    event_sink: JsonlEventSink | None = None
    run_dir: Path | None = None

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # directory name finalized after we get trace_id; we create a temp dir first
        run_dir = artifacts_dir / f"{scenario.name}-{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if record_events:
            event_sink = JsonlEventSink(run_dir / "events.jsonl")

    start = time.monotonic()
    assertion_results: list[AssertionResult] = []
    trace_id: str = ""

    try:
        with chaos_context(
            name=scenario.name,
            chaos=scenario.chaos,
            providers=scenario.providers,
            emit_events=False,
            event_sink=event_sink,
        ) as ctx:
            trace_id = ctx.session_id
            try:
                result = _run_maybe_await(_run_agent(scenario.agent, ctx))
                ctx.result = result
                # Capture agent output for debugging
                if result is not None:
                    ctx.agent_output = str(result)
            except Exception as e:
                # Preserve context/metrics but allow assertions to reason about expected failures.
                ctx.error = f"{type(e).__name__}: {e}"
                ctx.result = None

            ctx.elapsed_s = time.monotonic() - start
            for a in scenario.assertions:
                try:
                    ar = a(ctx)  # any callable(ctx) is allowed
                    if isinstance(ar, AssertionResult):
                        assertion_results.append(ar)
                    elif isinstance(ar, bool):
                        assertion_results.append(
                            AssertionResult(
                                name=getattr(
                                    a, "name", getattr(a, "__name__", "assertion")
                                ),
                                passed=ar,
                                message="",
                            )
                        )
                    else:
                        assertion_results.append(
                            AssertionResult(
                                name=getattr(
                                    a, "name", getattr(a, "__name__", "assertion")
                                ),
                                passed=False,
                                message="assertion must return AssertionResult or bool",
                                measured=type(ar).__name__,
                                expected="AssertionResult|bool",
                            )
                        )
                except Exception as e:
                    assertion_results.append(
                        AssertionResult(
                            name=getattr(
                                a, "name", getattr(a, "__name__", "assertion")
                            ),
                            passed=False,
                            message=f"assertion raised: {type(e).__name__}: {e}",
                        )
                    )

            error_allowed = any(
                bool(getattr(a, "allows_error", False)) for a in scenario.assertions
            )
            passed = all(r.passed for r in assertion_results) and (
                ctx.error is None or error_allowed
            )
            scorecard = {
                "trace_id": trace_id,
                "scenario": scenario.name,
                "passed": passed,
                "elapsed_s": ctx.elapsed_s,
                "error": ctx.error,
                "llm_calls_total": ctx.metrics.total_calls,
                "llm_calls_failed": sum(
                    1 for c in ctx.metrics.call_history if not c.get("success", True)
                ),
                "faults_injected_total": len(ctx.metrics.faults_injected),
                "avg_latency_s": ctx.metrics.avg_latency,
                "success_rate": ctx.metrics.success_rate,
                "avg_ttft_s": ctx.metrics.avg_ttft,
            }
        # Store ctx values before exiting the with block
        agent_input = ctx.agent_input
        agent_output = ctx.agent_output
    except Exception as e:
        # Only errors outside the chaos_context setup/teardown land here.
        elapsed_s = time.monotonic() - start
        error = f"{type(e).__name__}: {e}"
        passed = False
        agent_input = None
        agent_output = None
        scorecard = {
            "trace_id": trace_id,
            "scenario": scenario.name,
            "passed": False,
            "elapsed_s": elapsed_s,
            "error": error,
        }

    report = RunReport(
        scenario_name=scenario.name,
        trace_id=trace_id,
        passed=passed,
        elapsed_s=scorecard.get("elapsed_s") or 0.0,
        assertion_results=assertion_results,
        error=scorecard.get("error"),
        scorecard=scorecard,
        meta=scenario.meta,
        agent_input=agent_input,
        agent_output=agent_output,
    )

    if run_dir is not None:
        (run_dir / "scorecard.json").write_text(report.to_json(), encoding="utf-8")

        # If we have a trace_id, rename run dir to include it for convenience.
        if trace_id:
            final_dir = (
                run_dir.parent / f"{scenario.name}-{trace_id}-{int(time.time())}"
            )
            if final_dir != run_dir:
                try:
                    run_dir.rename(final_dir)
                except Exception:
                    pass

    return report
