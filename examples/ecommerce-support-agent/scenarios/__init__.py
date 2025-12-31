"""E-commerce Support Agent Scenarios.

This package provides chaos engineering scenarios for testing an
e-commerce customer support agent.

**Quick Start** (~3 min):
    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/quickstart.py

**Resilience Testing** (~10 min):
    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/resilience.py

**Automated Fuzzing** (~15 min):
    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/fuzzing.py

**All Scenarios**:
    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/

Structure:
- baselines.py: Shared baseline experiments (customer_journey, frustrated_customer)
- quickstart.py: First chaos tests - see the basics
- resilience.py: Production failure modes - LLM, tool, user input resilience
- fuzzing.py: Automated chaos discovery with ChaosSpace
"""

from .quickstart import get_scenarios as get_quickstart_scenarios
from .resilience import get_scenarios as get_resilience_scenarios
from .fuzzing import get_scenarios as get_fuzzing_scenarios


def get_scenarios():
    """Return all scenarios (quickstart + resilience + fuzzing)."""
    return (
        get_quickstart_scenarios()
        + get_resilience_scenarios()
        + get_fuzzing_scenarios()
    )
