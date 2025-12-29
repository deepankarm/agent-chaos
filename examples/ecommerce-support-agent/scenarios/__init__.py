"""E-commerce Support Agent Chaos Scenarios.

These scenarios demonstrate agent-chaos capabilities for testing
production AI agents in realistic failure modes.
"""

from .baseline import baseline_scenarios
from .deepeval_quality import deepeval_scenarios

# from .llm_failures import llm_failure_scenarios
# from .multi_turn import multi_turn_scenarios
# from .semantic_attacks import semantic_attack_scenarios
# from .tool_failures import tool_failure_scenarios


def get_scenarios():
    scenarios = (
        # baseline_scenarios
        # + tool_failure_scenarios
        # + llm_failure_scenarios
        # + multi_turn_scenarios
        # + semantic_attack_scenarios
        deepeval_scenarios
    )

    return scenarios
