from __future__ import annotations

import asyncio
import inspect
import random
import time
from pathlib import Path
from typing import Any

from agent_chaos.core.context import ChaosContext, chaos_context
from agent_chaos.event.jsonl import JsonlEventSink
from agent_chaos.scenario.assertions import AssertionResult
from agent_chaos.scenario.model import Scenario, Turn
from agent_chaos.scenario.report import RunReport


def _run_maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _run_turns(
    scenario: Scenario,
    ctx: ChaosContext,
    scenario_assertion_results: list[AssertionResult],
) -> None:
    """Run scenario turns.

    Executes each turn in sequence, tracking results and running
    turn-scoped assertions after each turn.

    Turn-level assertions are stored in each TurnResult.
    Scenario-level assertions go into scenario_assertion_results.
    """
    all_responses: list[str] = []

    for turn_number, turn in enumerate(scenario.turns, start=1):
        # Get turn input (static or dynamic)
        turn_input = turn.get_input(ctx.turn_results)
        ctx._current_turn_dynamic = turn.is_dynamic()

        # Store first turn input as agent_input for backward compat
        if turn_number == 1:
            ctx.agent_input = turn_input

        # Start the turn (records turn_start entry)
        ctx.start_turn(turn_number, turn_input)

        # Record user message AFTER turn_start
        ctx.metrics.add_conversation_entry(
            "user",
            content=turn_input,
            turn_number=turn_number,
            is_dynamic=turn.is_dynamic(),
        )

        # Track turn-scoped chaos config before registering
        turn_chaos_config = _get_turn_chaos_config(turn)

        # Register turn-scoped chaos with injector
        # (These will be active only for this turn due to on_turn trigger)
        _register_turn_chaos(ctx, turn, turn_number)

        # Run the agent for this turn
        turn_error: str | None = None
        response: str = ""
        try:
            result = _run_maybe_await(scenario.agent(ctx, turn_input))
            response = str(result) if result is not None else ""
            all_responses.append(response)

            # Record assistant response
            ctx.metrics.add_conversation_entry(
                "assistant",
                content=response,
                turn_number=turn_number,
            )
        except Exception as e:
            turn_error = f"{type(e).__name__}: {e}"
            response = ""

        # End the turn (this creates the TurnResult)
        ctx.end_turn(
            turn_input=turn_input,
            response=response,
            success=(turn_error is None),
            error=turn_error,
        )

        # Mark turn complete in injector
        ctx.injector.complete_turn()

        # Run turn-scoped assertions and store in TurnResult
        turn_assertion_results: list[AssertionResult] = []
        _run_assertions(turn.assertions, ctx, turn_assertion_results, turn_number)

        # Update the TurnResult with chaos and assertions
        if ctx.turn_results:
            current_turn_result = ctx.turn_results[-1]
            current_turn_result.chaos = turn_chaos_config
            current_turn_result.assertion_results = turn_assertion_results
            current_turn_result.is_dynamic = turn.is_dynamic()

        # Propagate first error to ctx.error for backward compat
        if turn_error and ctx.error is None:
            ctx.error = turn_error

    # Set agent_output to the last response for backward compat
    if all_responses:
        ctx.agent_output = all_responses[-1]
    ctx.result = all_responses


def _extract_chaos_config(chaos: Any) -> dict[str, Any]:
    """Extract JSON-serializable config from a chaos object or builder."""
    from agent_chaos.chaos.builder import ChaosBuilder

    if isinstance(chaos, ChaosBuilder):
        cfg = chaos._config
        # Derive type from chaos class name (e.g., ToolErrorChaos -> tool_error)
        class_name = chaos._chaos_class.__name__
        chaos_type = _class_name_to_type(class_name)

        # Get mutator function info if present
        mutator = cfg.get("mutator")
        fn_name = getattr(mutator, "__name__", None) if mutator else None
        fn_doc = getattr(mutator, "__doc__", None) if mutator else None
        # Truncate docstring to first line
        if fn_doc:
            fn_doc = fn_doc.strip().split("\n")[0]

        return {
            "chaos_type": chaos_type,
            "target_tool": cfg.get("tool_name"),
            "message": cfg.get("message"),
            "chaos_fn_name": fn_name,
            "chaos_fn_doc": fn_doc,
            "on_turn": cfg.get("on_turn"),
            "after_calls": cfg.get("after_calls"),
        }
    else:
        # Built Chaos object - use str() representation and attributes
        chaos_type = _class_name_to_type(chaos.__class__.__name__)
        return {
            "chaos_type": chaos_type,
            "point": str(getattr(chaos, "point", "")),
            "target_tool": getattr(chaos, "tool_name", None),
            "message": getattr(chaos, "message", None),
            "chaos_fn_name": None,
            "chaos_fn_doc": None,
            "on_turn": getattr(chaos, "on_turn", None),
            "after_calls": getattr(chaos, "after_calls", None),
        }


def _class_name_to_type(class_name: str) -> str:
    """Convert class name to readable type (e.g., ToolErrorChaos -> tool_error)."""
    # Remove 'Chaos' suffix
    name = class_name.replace("Chaos", "")
    # Convert CamelCase to snake_case
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _get_turn_chaos_config(turn: Turn) -> list[dict[str, Any]]:
    """Extract chaos configuration from a turn for UI display."""
    return [_extract_chaos_config(chaos) for chaos in turn.chaos]


def _get_scenario_chaos_config(scenario: Scenario) -> list[dict[str, Any]]:
    """Extract chaos configuration from a scenario for UI display."""
    return [_extract_chaos_config(chaos) for chaos in scenario.chaos]


def _register_turn_chaos(ctx: ChaosContext, turn: Turn, turn_number: int) -> None:
    """Register turn-scoped chaos with the injector.

    Turn-scoped chaos is automatically scoped to the current turn
    by adding an on_turn trigger if not already specified.
    """
    from agent_chaos.chaos.builder import ChaosBuilder

    for chaos in turn.chaos:
        if isinstance(chaos, ChaosBuilder):
            # Add on_turn trigger if not already set
            if chaos._config.get("on_turn") is None:
                chaos._config["on_turn"] = turn_number
            built = chaos.build()
        else:
            built = chaos

        # Add to appropriate list in injector based on point
        point = built.point
        from agent_chaos.chaos.base import ChaosPoint

        if point == ChaosPoint.USER_INPUT:
            ctx.injector._user_chaos.append(built)
        elif point == ChaosPoint.LLM_CALL:
            ctx.injector._llm_chaos.append(built)
        elif point == ChaosPoint.STREAM:
            ctx.injector._stream_chaos.append(built)
        elif point == ChaosPoint.TOOL_RESULT:
            ctx.injector._tool_chaos.append(built)
        elif point == ChaosPoint.MESSAGES:
            ctx.injector._context_chaos.append(built)


def _run_assertions(
    assertions: list[Any],
    ctx: ChaosContext,
    results: list[AssertionResult],
    turn_number: int | None = None,
) -> None:
    """Run assertions and collect results.

    Args:
        assertions: List of assertion callables.
        ctx: The chaos context.
        results: List to append results to.
        turn_number: If provided, pass to assertions that accept it.
    """
    for a in assertions:
        try:
            # Try calling with turn_number if assertion accepts it
            sig = inspect.signature(a.__call__ if hasattr(a, "__call__") else a)
            params = list(sig.parameters.keys())

            if turn_number is not None and "turn_number" in params:
                ar = a(ctx, turn_number=turn_number)
            else:
                ar = a(ctx)

            if isinstance(ar, AssertionResult):
                results.append(ar)
            elif isinstance(ar, bool):
                results.append(
                    AssertionResult(
                        name=getattr(a, "name", getattr(a, "__name__", "assertion")),
                        passed=ar,
                        message="",
                    )
                )
            else:
                results.append(
                    AssertionResult(
                        name=getattr(a, "name", getattr(a, "__name__", "assertion")),
                        passed=False,
                        message="assertion must return AssertionResult or bool",
                        measured=type(ar).__name__,
                        expected="AssertionResult|bool",
                    )
                )
        except Exception as e:
            results.append(
                AssertionResult(
                    name=getattr(a, "name", getattr(a, "__name__", "assertion")),
                    passed=False,
                    message=f"assertion raised: {type(e).__name__}: {e}",
                )
            )


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
            description=scenario.description,
            chaos=scenario.chaos,
            providers=scenario.providers,
            emit_events=False,
            event_sink=event_sink,
        ) as ctx:
            trace_id = ctx.session_id

            # Run all turns
            _run_turns(scenario, ctx, assertion_results)

            ctx.elapsed_s = time.monotonic() - start

            # Run scenario-level assertions
            _run_assertions(scenario.assertions, ctx, assertion_results)

            error_allowed = any(
                bool(getattr(a, "allows_error", False)) for a in scenario.assertions
            )
            # Check scenario-level assertions
            scenario_passed = all(r.passed for r in assertion_results)
            # Check turn-level assertions
            turn_passed = all(
                ar.passed
                for tr in ctx.turn_results
                for ar in getattr(tr, "assertion_results", [])
            )
            passed = (
                scenario_passed and turn_passed and (ctx.error is None or error_allowed)
            )
            # Get scenario-level chaos config for UI
            scenario_chaos_config = _get_scenario_chaos_config(scenario)

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
                # Turn stats
                "turns_total": len(scenario.turns),
                "turns_completed": len(ctx.turn_results),
                "turns_failed": sum(1 for tr in ctx.turn_results if not tr.success),
                # Scenario-level chaos config (for UI)
                "scenario_chaos": scenario_chaos_config,
            }
        # Store ctx values before exiting the with block
        agent_input = ctx.agent_input
        agent_output = ctx.agent_output
        conversation = ctx.metrics.conversation.copy()
        turn_results_data = [
            {
                "turn_number": tr.turn_number,
                "input": tr.input,
                "response": tr.response,
                "success": tr.success,
                "duration_s": tr.duration_s,
                "llm_calls": tr.llm_calls,
                "error": tr.error,
                "is_dynamic": getattr(tr, "is_dynamic", False),
                "chaos": getattr(tr, "chaos", []),
                "assertion_results": [
                    {
                        "name": ar.name,
                        "passed": ar.passed,
                        "message": ar.message,
                        "measured": ar.measured,
                        "expected": ar.expected,
                    }
                    for ar in getattr(tr, "assertion_results", [])
                ],
            }
            for tr in ctx.turn_results
        ]

    except Exception as e:
        # Only errors outside the chaos_context setup/teardown land here.
        elapsed_s = time.monotonic() - start
        error = f"{type(e).__name__}: {e}"
        passed = False
        agent_input = None
        agent_output = None
        conversation = []
        turn_results_data = []
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
        description=scenario.description,
        assertion_results=assertion_results,
        error=scorecard.get("error"),
        scorecard=scorecard,
        tags=scenario.tags,
        agent_input=agent_input,
        agent_output=agent_output,
        conversation=conversation,
        turn_results=turn_results_data,
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
