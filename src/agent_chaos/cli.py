from __future__ import annotations

import argparse
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_chaos")


def _get_scenario_label(scenario: Any) -> str:
    """Get a label describing the scenario type (baseline or variant).

    Args:
        scenario: A BaselineScenario or ChaosScenario instance.

    Returns:
        A string like "baseline" or "← parent-name".
    """
    from agent_chaos.scenario.model import ChaosScenario

    if isinstance(scenario, ChaosScenario) and scenario.parent:
        return f"← {scenario.parent}"
    return "baseline"


def _setup_logging() -> None:
    """Configure logging for agent-chaos CLI."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _suppress_event_loop_closed_errors() -> None:
    """Suppress 'Event loop is closed' errors from httpx client cleanup.

    Libraries like pydantic-ai and anthropic create httpx.AsyncClient instances
    internally. When asyncio.run() closes the event loop, these clients try to
    clean up their connections but the loop is already closed, causing noisy
    (but harmless) RuntimeError exceptions.
    """

    class EventLoopClosedFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "Event loop is closed" in msg:
                return False
            if record.exc_info:
                exc = record.exc_info[1]
                if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                    return False
            return True

    logging.getLogger("asyncio").addFilter(EventLoopClosedFilter())


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent-chaos",
        description="Chaos engineering harness for AI agents (CLI-first).",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # --- run command ---
    run_p = sub.add_parser(
        "run",
        help="Run one or more scenarios (files, module:attr, or directories)",
    )
    run_p.add_argument(
        "targets",
        nargs="+",
        help="Targets: path/to/file.py OR package.module:attr OR directory",
    )
    run_p.add_argument(
        "--glob",
        default="*.py",
        help="When a target is a directory, discover scenarios using this glob (default: *.py)",
    )
    run_p.add_argument(
        "--recursive",
        action="store_true",
        help="When a target is a directory, search recursively for matching files",
    )
    run_p.add_argument(
        "--artifacts-dir",
        default=".agent_chaos_runs",
        help="Directory where run artifacts are written (default: .agent_chaos_runs)",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for probabilistic faults (optional)",
    )
    run_p.add_argument(
        "--no-events",
        action="store_true",
        help="Do not write events.jsonl (still writes scorecard.json)",
    )
    run_p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failing scenario",
    )
    run_p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: 0 = CPU count). Use 1 for sequential.",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenario details without running them",
    )

    # --- ui command ---
    ui_p = sub.add_parser("ui", help="Start the dashboard server")
    ui_p.add_argument(
        "runs_dir",
        help="Directory containing run artifacts from 'agent-chaos run'",
    )
    ui_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    ui_p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )

    return parser


def _run_scenario_worker(
    args: tuple[str, int, str, str, int | None, bool],
) -> dict[str, Any]:
    """Worker function for parallel scenario execution.

    Runs in a subprocess. Loads scenario fresh and executes it.

    Args:
        args: Tuple of (source_ref, source_index, scenario_name, artifacts_dir, seed, record_events)

    Returns:
        Dict with scenario results (serializable subset of RunReport).
    """
    source_ref, source_index, scenario_name, artifacts_dir, seed, record_events = args

    _suppress_event_loop_closed_errors()

    try:
        from agent_chaos.scenario.loader import load_scenario_by_index
        from agent_chaos.scenario.runner import run_scenario

        scenario = load_scenario_by_index(source_ref, source_index)
        report = run_scenario(
            scenario,
            artifacts_dir=Path(artifacts_dir),
            seed=seed,
            record_events=record_events,
        )

        return {
            "scenario_name": report.scenario_name,
            "passed": report.passed,
            "elapsed_s": report.elapsed_s,
            "error": report.error,
            "assertion_results": [
                {"name": ar.name, "passed": ar.passed, "message": ar.message}
                for ar in report.assertion_results
            ],
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Worker failed for {scenario_name} (source: {source_ref}:{source_index})\n{tb}"
        ) from e


def _cmd_ui(args: argparse.Namespace) -> None:
    """Handle the 'ui' command."""
    from agent_chaos.ui.server import run_server

    run_server(runs_dir=Path(args.runs_dir), host=args.host, port=args.port)


def _cmd_run_dry(scenarios: list) -> None:
    """Handle --dry-run mode: print scenario details without running."""
    from agent_chaos.scenario.model import ChaosScenario

    total = len(scenarios)
    logger.info(f"\n🃏 Found {total} scenario(s):\n")

    baselines = [
        s for s in scenarios if not isinstance(s, ChaosScenario) or not s.parent
    ]

    for scenario in scenarios:
        label = _get_scenario_label(scenario)
        chaos_count = len(scenario.chaos) if isinstance(scenario, ChaosScenario) else 0
        turns_count = len(scenario.turns)

        logger.info(f"  {scenario.name} ({label})")
        logger.info(f"    {scenario.description}")
        logger.info(
            f"    turns: {turns_count}, chaos: {chaos_count}, assertions: {len(scenario.assertions)}"
        )

        # Show chaos details for variants
        if isinstance(scenario, ChaosScenario) and scenario.chaos:
            for chaos in scenario.chaos[:3]:
                chaos_str = (
                    str(chaos.build()) if hasattr(chaos, "build") else str(chaos)
                )
                logger.info(f"      • {chaos_str}")
            if len(scenario.chaos) > 3:
                logger.info(f"      ... and {len(scenario.chaos) - 3} more")

        logger.info("")

    baseline_count = len(baselines)
    variant_count = total - baseline_count
    logger.info(f"Summary: {baseline_count} baseline(s), {variant_count} variant(s)")


def _cmd_run_sequential(
    scenarios: list,
    artifacts_dir: Path,
    seed: int | None,
    record_events: bool,
    fail_fast: bool,
) -> tuple[int, int]:
    """Run scenarios sequentially.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    from agent_chaos.scenario.runner import run_scenario

    passed = 0
    failed = 0
    total = len(scenarios)

    for i, scenario in enumerate(scenarios, 1):
        label = _get_scenario_label(scenario)
        logger.info(f"[{i}/{total}] {scenario.name} ({label})...")

        report = run_scenario(
            scenario,
            artifacts_dir=artifacts_dir,
            seed=seed,
            record_events=record_events,
        )

        if report.passed:
            passed += 1
            logger.info(f"  ✓ PASS ({report.elapsed_s:.2f}s)")
        else:
            failed += 1
            logger.info(f"  ✗ FAIL ({report.elapsed_s:.2f}s)")
            if report.error:
                logger.info(f"    Error: {report.error}")
            for ar in report.assertion_results:
                if not ar.passed:
                    logger.info(f"    • {ar.name}: {ar.message}")
            if fail_fast:
                break

    return passed, failed


def _cmd_run_parallel(
    scenarios: list,
    artifacts_dir: Path,
    seed: int | None,
    record_events: bool,
    workers: int,
) -> tuple[int, int]:
    """Run scenarios in parallel using ProcessPoolExecutor.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    passed = 0
    failed = 0

    work_items = [
        (
            getattr(s, "_source_ref", "unknown"),
            getattr(s, "_source_index", 0),
            s.name,
            str(artifacts_dir),
            seed,
            record_events,
        )
        for s in scenarios
    ]

    scenario_labels = {s.name: _get_scenario_label(s) for s in scenarios}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_name = {}
        for item in work_items:
            scenario_name = item[2]
            label = scenario_labels.get(scenario_name, "")
            logger.info(f"⏳ STARTING {scenario_name} ({label})")
            future = executor.submit(_run_scenario_worker, item)
            future_to_name[future] = scenario_name

        for future in as_completed(future_to_name):
            scenario_name = future_to_name[future]
            try:
                result = future.result()
                if result["passed"]:
                    passed += 1
                    logger.info(
                        f"✓ PASS {result['scenario_name']} ({result['elapsed_s']:.2f}s)"
                    )
                else:
                    failed += 1
                    logger.info(
                        f"✗ FAIL {result['scenario_name']} ({result['elapsed_s']:.2f}s)"
                    )
                    if result["error"]:
                        logger.info(f"    Error: {result['error']}")
                    for ar in result["assertion_results"]:
                        if not ar["passed"]:
                            logger.info(f"    • {ar['name']}: {ar['message']}")
            except Exception as e:
                failed += 1
                error_type = type(e).__name__
                error_msg = str(e) or "(no message)"
                logger.error(
                    f"✗ FAIL {scenario_name} (worker error: {error_type}: {error_msg})"
                )
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                for line in tb.strip().split("\n"):
                    logger.error(f"    {line}")

    return passed, failed


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' command."""
    _setup_logging()

    from agent_chaos.scenario.loader import load_scenarios

    scenarios = load_scenarios(args.targets, glob=args.glob, recursive=args.recursive)
    total = len(scenarios)

    # Dry-run mode
    if args.dry_run:
        _cmd_run_dry(scenarios)
        return

    # Resolve worker count
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = min(workers, total)

    logger.info(f"\n🃏 Running {total} scenario(s) with {workers} worker(s)...\n")

    artifacts_dir = Path(args.artifacts_dir)
    record_events = not args.no_events

    if workers == 1:
        passed, failed = _cmd_run_sequential(
            scenarios, artifacts_dir, args.seed, record_events, args.fail_fast
        )
    else:
        passed, failed = _cmd_run_parallel(
            scenarios, artifacts_dir, args.seed, record_events, workers
        )

    logger.info("")
    logger.info(f"Results: {passed} passed, {failed} failed, {total} total")
    raise SystemExit(0 if failed == 0 else 1)


def main() -> None:
    """Console script entry point (`agent-chaos`)."""
    _suppress_event_loop_closed_errors()

    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "ui":
        _cmd_ui(args)
    elif args.cmd == "run":
        _cmd_run(args)
    else:
        parser.print_help()
