"""Scenario model (Python-first).

Scenario files are just Python modules exposing `scenario: Scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_chaos.chaos.base import Chaos
from agent_chaos.chaos.builder import ChaosBuilder


@dataclass
class Scenario:
    """A single chaos scenario.

    Attributes:
        name: Unique scenario name.
        description: Human-readable description of what this scenario tests.
        agent: Callable that runs the agent under test.
            - If it accepts 'ctx' argument, it will be passed the ChaosContext.
            - Otherwise it will be called with no arguments.
        chaos: List of chaos to inject.
        providers: Providers to patch. Defaults to ["anthropic"].
        assertions: List of assertions to validate contracts.
        meta: Optional metadata for CI/reporting (model, commit SHA, etc.)

    Example:
        from agent_chaos import llm_rate_limit, llm_stream_cut
        from agent_chaos.scenario import Scenario, CompletesWithin

        scenario = Scenario(
            name="rate-limit-recovery",
            description="Tests agent resilience to LLM rate limit errors",
            agent=my_driver,
            chaos=[
                llm_rate_limit().after_calls(2),
                llm_stream_cut(after_chunks=10),
            ],
            assertions=[CompletesWithin(timeout_s=30.0)],
        )
    """

    name: str
    description: str
    agent: Callable[..., Any]
    chaos: list[Chaos | ChaosBuilder] = field(default_factory=list)
    providers: list[str] = field(default_factory=lambda: ["anthropic"])
    assertions: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
