from __future__ import annotations

import argparse
import logging
from pathlib import Path


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

    from agent_chaos.scenario.loader import load_scenarios
    from agent_chaos.scenario.runner import run_scenario

    scenarios = load_scenarios(args.targets, glob=args.glob, recursive=args.recursive)
    print(f"\n🃏 Running {len(scenarios)} scenario(s)...\n")

    passed = 0
    failed = 0

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)
        report = run_scenario(
            scenario,
            artifacts_dir=Path(args.artifacts_dir),
            seed=args.seed,
            record_events=not args.no_events,
        )

        if report.passed:
            passed += 1
            print(f"✓ PASS ({report.elapsed_s:.2f}s)")
        else:
            failed += 1
            print(f"✗ FAIL ({report.elapsed_s:.2f}s)")
            if report.error:
                print(f"    Error: {report.error}")
            for ar in report.assertion_results:
                if not ar.passed:
                    print(f"    • {ar.name}: {ar.message}")
            if args.fail_fast:
                break

    summary = {
        "scenarios_total": len(scenarios),
        "passed": passed,
        "failed": failed,
    }
    print(summary)
    raise SystemExit(0 if failed == 0 else 1)
