"""Scenario run report + stable scorecard output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_chaos.scenario.assertions import AssertionResult


@dataclass
class RunReport:
    scenario_name: str
    trace_id: str
    passed: bool
    elapsed_s: float
    description: str = ""
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    scorecard: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    agent_input: str | None = None
    agent_output: str | None = None
    conversation: list[dict[str, Any]] = field(default_factory=list)
    turn_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["assertion_results"] = [asdict(r) for r in self.assertion_results]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
