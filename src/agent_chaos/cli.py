"""CLI entrypoint for agent-chaos.

Keep CLI code out of `agent_chaos/__init__.py` so importing the library stays clean.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Console script entry point (`agent-chaos`)."""
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

    args = parser.parse_args()

    if args.cmd != "run":
        parser.print_help()
        return

    from agent_chaos.scenario.loader import load_scenarios_from_dir, load_target
    from agent_chaos.scenario.runner import run_scenario

    scenarios = []
    for t in args.targets:
        p = Path(t)
        if p.exists() and p.is_dir():
            scenarios.extend(
                load_scenarios_from_dir(p, glob=args.glob, recursive=args.recursive)
            )
        else:
            scenarios.extend(load_target(t))

    passed = 0
    failed = 0

    for scenario in scenarios:
        report = run_scenario(
            scenario,
            artifacts_dir=Path(args.artifacts_dir),
            seed=args.seed,
            record_events=not args.no_events,
        )
        if report.passed:
            passed += 1
        else:
            failed += 1
            if args.fail_fast:
                break

    summary = {
        "scenarios_total": len(scenarios),
        "passed": passed,
        "failed": failed,
    }
    print(summary)
    raise SystemExit(0 if failed == 0 else 1)
