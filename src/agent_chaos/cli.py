from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_chaos")


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

    This suppresses asyncio's "Task exception was never retrieved" log messages
    when the event loop is closed.
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

    # Suppress asyncio errors in worker too
    _suppress_event_loop_closed_errors()

    from agent_chaos.scenario.loader import load_scenario_by_index
    from agent_chaos.scenario.runner import run_scenario

    scenario = load_scenario_by_index(source_ref, source_index)
    report = run_scenario(
        scenario,
        artifacts_dir=Path(artifacts_dir),
        seed=seed,
        record_events=record_events,
    )

    # Return serializable dict (RunReport may not be pickleable)
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


def main() -> None:
    """Console script entry point (`agent-chaos`)."""
    _suppress_event_loop_closed_errors()

    parser = argparse.ArgumentParser(
        prog="agent-chaos",
        description="Chaos engineering harness for AI agents (CLI-first).",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

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

    args = parser.parse_args()

    if args.cmd == "ui":
        from agent_chaos.ui.server import run_server

        run_server(runs_dir=Path(args.runs_dir), host=args.host, port=args.port)
        return

    if args.cmd != "run":
        parser.print_help()
        return

    _setup_logging()

    from agent_chaos.scenario.loader import load_scenarios
    from agent_chaos.scenario.runner import run_scenario

    scenarios = load_scenarios(args.targets, glob=args.glob, recursive=args.recursive)
    total = len(scenarios)

    # Resolve worker count
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = min(workers, total)  # No point having more workers than scenarios

    logger.info(f"\n🃏 Running {total} scenario(s) with {workers} worker(s)...\n")

    passed = 0
    failed = 0
    artifacts_dir = Path(args.artifacts_dir)

    if workers == 1:
        # Sequential execution (original behavior)
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"[{i}/{total}] {scenario.name}...")
            report = run_scenario(
                scenario,
                artifacts_dir=artifacts_dir,
                seed=args.seed,
                record_events=not args.no_events,
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
                if args.fail_fast:
                    break
    else:
        # Parallel execution
        work_items = [
            (
                getattr(s, "_source_ref", "unknown"),
                getattr(s, "_source_index", 0),
                s.name,
                str(artifacts_dir),
                args.seed,
                not args.no_events,
            )
            for s in scenarios
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_name = {}
            for item in work_items:
                logger.info(f"⏳ STARTING {item[2]}")
                future = executor.submit(_run_scenario_worker, item)
                future_to_name[future] = item[2]

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
                    logger.error(f"✗ FAIL {scenario_name} (worker error: {e})")

    logger.info("")
    logger.info(f"Results: {passed} passed, {failed} failed, {total} total")
    raise SystemExit(0 if failed == 0 else 1)
