"""Quickstart scenarios - Your first chaos tests (~3 minutes).

This module demonstrates the core capabilities of agent-chaos:

1. **Baselines**: See the agent work without chaos
2. **LLM Chaos**: What happens when the LLM provider has issues?
3. **Tool Chaos**: What happens when backend services fail?
4. **Quality Gates**: Semantic evaluation with LLM-as-judge

Run these first to understand what agent-chaos can do:

    uv run agent-chaos run examples/ecommerce-support-agent/scenarios/quickstart.py
    uv run agent-chaos ui

Each scenario builds on the same 5-turn customer journey, making it easy
to compare results across different failure conditions.
"""

from agent_chaos import at
from agent_chaos.chaos import llm_rate_limit, tool_error
from agent_chaos.scenario import MaxTotalLLMCalls, MinChaosInjected, Scenario

from .baselines import customer_journey, frustrated_customer
from .commons import error_handling, task_completion


def get_scenarios() -> list[Scenario]:
    """Return quickstart scenarios."""
    return [
        # ---------------------------------------------------------------------
        # 1. The Happy Path
        # See the agent work without any chaos. This is your control group.
        # ---------------------------------------------------------------------
        customer_journey,
        # ---------------------------------------------------------------------
        # 2. LLM Rate Limit
        # LLM returns 429 after the first call. Does the agent retry gracefully
        # or crash? Does the user see a helpful message or raw error?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="llm-rate-limit",
            description="LLM returns 429 after first call - does agent recover?",
            chaos=[llm_rate_limit().after_calls(1)],
            assertions=[MinChaosInjected(1)],
            tags=["quickstart", "llm"],
        ),
        # ---------------------------------------------------------------------
        # 3. Tool Error on Refund Check
        # The refund service fails when the customer asks about refunds (turn 2).
        # Critical question: Does the agent LIE about processing the refund,
        # or does it gracefully acknowledge the failure?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="tool-error-refund",
            description="Refund service fails - does agent lie about processing?",
            turns=[
                at(
                    2,
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
        # ---------------------------------------------------------------------
        # 4. Quality Gate
        # Same journey, but with semantic quality evaluation.
        # Did the agent complete tasks correctly? Did it handle issues well?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="quality-evaluation",
            description="Semantic quality check - did agent handle everything correctly?",
            assertions=[task_completion, error_handling],
            tags=["quickstart", "quality"],
        ),
        # ---------------------------------------------------------------------
        # 5. Frustrated Customer
        # Dynamic LLM-generated customer escalation. Each turn gets more
        # frustrated based on the agent's previous response.
        # Tests: Does agent maintain professionalism? Does it escalate?
        # ---------------------------------------------------------------------
        frustrated_customer,
        # ---------------------------------------------------------------------
        # 6. Combined Chaos + Quality
        # LLM issues AND tool failures AND quality evaluation.
        # The real test: Can the agent survive multiple failures AND
        # still provide a quality experience?
        # ---------------------------------------------------------------------
        customer_journey.variant(
            name="combined-chaos-quality",
            description="Multiple failures with semantic quality validation",
            chaos=[llm_rate_limit().after_calls(3)],
            turns=[
                at(
                    2,
                    chaos=[
                        tool_error("Connection timeout").for_tool(
                            "check_refund_eligibility"
                        )
                    ],
                ),
            ],
            assertions=[
                MinChaosInjected(1),
                MaxTotalLLMCalls(20),
                task_completion,
                error_handling,
            ],
            tags=["quickstart", "combined"],
        ),
    ]
