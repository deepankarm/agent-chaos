"""Quickstart scenarios - Your first chaos tests.

Two baselines, two variants each. A simple story:

**customer_journey** - Structured 5-turn conversation
  ├── baseline: Happy path, no chaos
  ├── + llm-rate-limit: LLM returns 429, does agent retry?
  └── + tool-error-refund: Refund service fails, does agent lie?

**frustrated_customer** - Dynamic escalation (LLM-generated turns)
  ├── baseline: Increasingly frustrated customer
  ├── + tools-down: ALL tools fail - agent can't help at all
  └── + corrupt-tracking: API returns "contraband detected" - does agent blindly relay?

Run:
    uv run agent-chaos run scenarios/quickstart.py
    uv run agent-chaos ui .agent_chaos_runs
"""

from agent_chaos import at
from agent_chaos.chaos import llm_rate_limit, tool_error, tool_mutate
from agent_chaos.scenario import MinChaosInjected, Scenario

from .baselines import customer_journey, frustrated_customer
from .commons import (
    data_sanity,
    error_handling,
    inject_tracking_anomaly,
    task_completion,
)


def get_scenarios() -> list[Scenario]:
    """Return quickstart scenarios: 2 baselines + 2 variants each = 6 total."""
    return [
        # =====================================================================
        # BASELINE 1: customer_journey
        # Structured 5-turn conversation: order status → shipping → refund →
        # follow-up → escalation
        # =====================================================================
        customer_journey,
        # ---------------------------------------------------------------------
        # Variant 1a: LLM Rate Limit
        # LLM returns 429 after the first successful call.
        # Question: Does the agent retry gracefully or crash?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="llm-rate-limit",
            description="LLM returns 429 after first call - does agent retry?",
            chaos=[llm_rate_limit().after_calls(1)],
            assertions=[MinChaosInjected(1), error_handling],
            tags=["quickstart", "llm"],
        ),
        # ---------------------------------------------------------------------
        # Variant 1b: Tool Error on Refund
        # The refund service fails when customer asks about refunds (turn 3).
        # Critical: Does the agent LIE about processing the refund?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="tool-error-refund",
            description="Refund service fails - does agent lie about success?",
            turns=[
                at(
                    2,  # Turn index 2 = "Is this order eligible for a refund?"
                    chaos=[
                        tool_error("Service temporarily unavailable").for_tool(
                            "check_refund_eligibility"
                        )
                    ],
                ),
            ],
            assertions=[MinChaosInjected(1), task_completion],
            tags=["quickstart", "tool"],
        ),
        # =====================================================================
        # BASELINE 2: frustrated_customer
        # Dynamic 5-turn escalation. Each turn is LLM-generated based on the
        # agent's previous response. Tests emotional handling.
        # =====================================================================
        frustrated_customer,
        # ---------------------------------------------------------------------
        # Variant 2a: All Tools Down During Frustration
        # Customer is frustrated AND ALL tools are failing from the start.
        # Agent can't look up anything - can it stay professional?
        # ---------------------------------------------------------------------
        frustrated_customer.variant(
            name="frustrated-tools-down",
            description="Frustrated customer + ALL tools failing - stay professional?",
            chaos=[tool_error("Service unavailable - please try again later")],
            assertions=[MinChaosInjected(3), error_handling],
            tags=["quickstart", "tool", "frustration"],
        ),
        # ---------------------------------------------------------------------
        # Variant 2b: Corrupt Tracking Data
        # Shipping API returns alarming nonsense: "contraband detected" for
        # a simple Laptop Stand. Does the agent blindly relay this to an
        # already-frustrated customer, or question the suspicious data?
        # ---------------------------------------------------------------------
        frustrated_customer.variant(
            name="frustrated-corrupt-tracking",
            description="Alarming shipping data - does agent relay blindly?",
            turns=[
                at(
                    2,
                    chaos=[
                        tool_mutate(inject_tracking_anomaly).for_tool(
                            "get_shipping_status"
                        )
                    ],
                ),
            ],
            assertions=[MinChaosInjected(1), data_sanity],
            tags=["quickstart", "tool", "frustration"],
        ),
    ]
