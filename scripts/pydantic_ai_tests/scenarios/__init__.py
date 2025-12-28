"""Weather agent chaos scenarios.

Usage:
    cd scripts/pydantic_ai_tests
    uv run agent-chaos run scenarios/
"""

from .baseline import baseline_scenario
from .context_scenarios import context_scenarios
from .edge_scenarios import edge_case_scenarios
from .llm_scenarios import llm_scenarios
from .multi_scenarios import multi_chaos_scenarios
from .semantic_scenarios import semantic_chaos_scenarios
from .streaming_scenarios import streaming_scenarios
from .tool_scenarios import tool_scenarios
from .user_scenarios import user_input_chaos_scenarios

__all__ = [
    "baseline_scenario",
    "llm_scenarios",
    "tool_scenarios",
    "streaming_scenarios",
    "context_scenarios",
    "multi_chaos_scenarios",
    "edge_case_scenarios",
    "semantic_chaos_scenarios",
    "user_input_chaos_scenarios",
    "get_scenarios",
]


def get_scenarios():
    """Return all scenarios to run.

    Uncomment scenario groups as needed for testing.
    """
    return [
        baseline_scenario,
        *user_input_chaos_scenarios,
        *llm_scenarios,
        *tool_scenarios,
        *streaming_scenarios,
        *context_scenarios,
        *multi_chaos_scenarios,
        *edge_case_scenarios,
        *semantic_chaos_scenarios,
    ]
