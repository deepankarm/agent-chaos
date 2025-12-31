"""Automated chaos fuzzing - Discover failures you didn't think of.

Instead of manually writing every chaos scenario, let agent-chaos
generate combinations automatically using ChaosSpace configurations.

This module applies different fuzzing strategies to the baseline experiments:

**Default Fuzz**: Balanced LLM + Tool chaos
**Tool Heavy**: Focus on tool failures with custom mutators
**LLM Heavy**: Focus on LLM provider issues
**Stress Test**: Everything enabled at high probability
**Malformed**: Test parsing resilience with truncated/corrupted responses

Run after resilience to discover edge cases:

    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/fuzzing.py
    uv run agent-chaos ui

The fuzzer generates N variants of each baseline, each with a different
random combination of chaos events based on the ChaosSpace configuration.
"""

from agent_chaos import (
    ChaosSpace,
    LLMFuzzConfig,
    StreamFuzzConfig,
    ToolFuzzConfig,
    fuzz_chaos,
)
from agent_chaos.scenario import Scenario

from .baselines import customer_journey, frustrated_customer
from .commons import (
    SUPPORTED_TOOLS,
    corrupt_order_status,
    corrupt_refund_amount,
    inject_conflicting_data,
    inject_tracking_anomaly,
    return_malformed_json,
)

# =============================================================================
# ChaosSpace Configurations
# =============================================================================

# Default balanced fuzzing
default_space = ChaosSpace(
    llm=LLMFuzzConfig(probability=0.3),
    stream=StreamFuzzConfig.disabled(),
    tool=ToolFuzzConfig(probability=0.4, targets=SUPPORTED_TOOLS),
    min_per_scenario=2,
    max_per_scenario=4,
)

# Tool-heavy fuzzing with custom mutators
tool_heavy_space = ChaosSpace(
    llm=LLMFuzzConfig(probability=0.1),
    stream=StreamFuzzConfig.disabled(),
    tool=ToolFuzzConfig(
        probability=0.6,
        targets=SUPPORTED_TOOLS,
        mutators=[corrupt_order_status, corrupt_refund_amount, inject_tracking_anomaly],
        mutator_probability=0.3,
    ),
    min_per_scenario=2,
    max_per_scenario=5,
)

# LLM-focused fuzzing
llm_heavy_space = ChaosSpace(
    llm=LLMFuzzConfig.heavy(),
    stream=StreamFuzzConfig.disabled(),
    tool=ToolFuzzConfig.disabled(),
    min_per_scenario=2,
    max_per_scenario=4,
)

# Stress testing with everything enabled
stress_space = ChaosSpace(
    llm=LLMFuzzConfig.heavy(),
    stream=StreamFuzzConfig.disabled(),
    tool=ToolFuzzConfig(
        probability=0.5,
        targets=SUPPORTED_TOOLS,
        mutators=[
            corrupt_order_status,
            corrupt_refund_amount,
            inject_tracking_anomaly,
            inject_conflicting_data,
        ],
        mutator_probability=0.4,
    ),
    min_per_scenario=3,
    max_per_scenario=6,
)

# Malformed responses only
malformed_space = ChaosSpace(
    llm=LLMFuzzConfig.disabled(),
    stream=StreamFuzzConfig.disabled(),
    tool=ToolFuzzConfig(
        probability=0.0,
        targets=SUPPORTED_TOOLS,
        mutators=[return_malformed_json],
        mutator_probability=0.8,
    ),
    min_per_scenario=1,
    max_per_scenario=3,
)


# =============================================================================
# Generate Fuzzed Scenarios
# =============================================================================

# Each fuzz_chaos() call generates N variants with random chaos combinations
fuzz_default = fuzz_chaos(customer_journey, n=5, seed=42, space=default_space)
fuzz_tool_heavy = fuzz_chaos(customer_journey, n=5, seed=43, space=tool_heavy_space)
fuzz_llm_heavy = fuzz_chaos(customer_journey, n=3, seed=44, space=llm_heavy_space)
fuzz_stress = fuzz_chaos(frustrated_customer, n=3, seed=45, space=stress_space)
fuzz_malformed = fuzz_chaos(customer_journey, n=2, seed=46, space=malformed_space)


def get_scenarios() -> list[Scenario]:
    """Return all fuzzed scenarios."""
    return (
        fuzz_default + fuzz_tool_heavy + fuzz_llm_heavy + fuzz_stress + fuzz_malformed
    )
