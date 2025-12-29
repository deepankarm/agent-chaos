"""Scenario runner for agent-chaos (CLI-first).

This package provides:
- a Python-first Scenario model (callable-based)
- Multi-turn support with Turn and TurnResult
- small assertion library (contracts)
- runner that produces a stable RunReport + artifacts
"""

from agent_chaos.scenario.assertions import (
    # Scenario-level assertions
    CompletesWithin,
    ExpectError,
    MaxFailedCalls,
    MaxLLMCalls,
    MinChaosInjected,
    MinLLMCalls,
    # Turn-aware assertions
    AllTurnsComplete,
    MaxTotalLLMCalls,
    RecoveredAfterFailure,
    TurnCompletes,
    TurnCompletesWithin,
    TurnMaxLLMCalls,
    TurnResponseContains,
)
from agent_chaos.scenario.model import Scenario, Turn, TurnResult
from agent_chaos.scenario.report import RunReport
from agent_chaos.scenario.runner import run_scenario

__all__ = [
    # Core
    "Scenario",
    "Turn",
    "TurnResult",
    "RunReport",
    "run_scenario",
    # Scenario-level assertions
    "CompletesWithin",
    "MaxLLMCalls",
    "MaxFailedCalls",
    "MinLLMCalls",
    "MinChaosInjected",
    "ExpectError",
    # Turn-aware assertions
    "TurnCompletes",
    "TurnCompletesWithin",
    "TurnResponseContains",
    "TurnMaxLLMCalls",
    "AllTurnsComplete",
    "RecoveredAfterFailure",
    "MaxTotalLLMCalls",
]
