"""Scenario runner for agent-chaos (CLI-first).

This package provides:
- a Python-first Scenario model (callable-based)
- small assertion library (contracts)
- runner that produces a stable RunReport + artifacts
"""

from agent_chaos.scenario.assertions import (
    CompletesWithin,
    ExpectError,
    MaxFailedCalls,
    MaxLLMCalls,
    MinChaosInjected,
    MinLLMCalls,
)
from agent_chaos.scenario.model import Scenario
from agent_chaos.scenario.report import RunReport
from agent_chaos.scenario.runner import run_scenario

__all__ = [
    "Scenario",
    "RunReport",
    "run_scenario",
    "CompletesWithin",
    "MaxLLMCalls",
    "MaxFailedCalls",
    "MinLLMCalls",
    "MinChaosInjected",
    "ExpectError",
]
