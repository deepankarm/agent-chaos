from __future__ import annotations

from agent_chaos import Turn, TurnResult
from agent_chaos.chaos import llm_rate_limit, tool_error
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    MaxTotalLLMCalls,
    RecoveredAfterFailure,
    Scenario,
    TurnCompletes,
    TurnResponseContains,
)

from agent import run_support_agent

# =============================================================================
# Dynamic Input Generators
# =============================================================================


def escalate_if_unhappy(history: list[TurnResult]) -> str:
    """Escalate if previous response was unsatisfactory."""
    if not history:
        return "I have a problem with my order"

    last = history[-1]
    if "error" in last.response.lower() or "unable" in last.response.lower():
        return "This is unacceptable! I want to speak to a manager immediately!"
    elif "refund" in last.response.lower():
        return "How long will the refund take? I need the money urgently."
    return "Thank you for your help."


def follow_up_on_refund(history: list[TurnResult]) -> str:
    """Generate follow-up based on refund response."""
    if not history:
        return "I need a refund"

    last = history[-1]
    if "not eligible" in last.response.lower() or "cannot" in last.response.lower():
        return (
            "But the product was defective! This is not fair. I want to escalate this."
        )
    elif "pending" in last.response.lower() or "processing" in last.response.lower():
        return "When exactly will I receive my refund? I need a specific date."
    return "Okay, thanks for processing that."


def progressive_frustration(history: list[TurnResult]) -> str:
    """Customer gets progressively more frustrated."""
    turn_num = len(history) + 1
    frustration_levels = {
        1: "I need help with my order ORD-67890",
        2: "Why is this taking so long? Where is my package?",
        3: "This is ridiculous! I've been waiting for days!",
        4: "I demand a refund and compensation for my time!",
    }
    return frustration_levels.get(turn_num, "Just give me my money back!")


# =============================================================================
# Multi-Turn Scenarios
# =============================================================================

multi_turn_scenarios = [
    Scenario(
        name="multi-turn-order-inquiry",
        description="Customer asks about order, then requests tracking details",
        agent=run_support_agent,
        turns=[
            Turn("What's the status of my order ORD-67890?"),
            Turn("Can you give me more details about the shipping?"),
        ],
        chaos=[],
        assertions=[AllTurnsComplete(), CompletesWithin(120.0), MaxTotalLLMCalls(12)],
        tags=["multi_turn", "order_inquiry"],
    ),
    Scenario(
        name="multi-turn-refund-journey",
        description="Customer inquires about refund eligibility, then requests processing",
        agent=run_support_agent,
        turns=[
            Turn("Can I get a refund for order ORD-67890?"),
            Turn(
                "Yes, please process the refund. The laptop stand doesn't fit my desk."
            ),
        ],
        chaos=[],
        assertions=[AllTurnsComplete(), CompletesWithin(120.0)],
        tags=["multi_turn", "refund"],
    ),
    Scenario(
        name="multi-turn-refund-fails-turn2",
        description="Refund check succeeds but processing fails on turn 2",
        agent=run_support_agent,
        turns=[
            Turn("Is order ORD-67890 eligible for a refund?"),
            Turn(
                input="Great, please process the refund now.",
                chaos=[
                    tool_error("Payment processor unavailable").for_tool(
                        "process_refund"
                    )
                ],
            ),
        ],
        assertions=[
            TurnCompletes(turn=1),
            AllTurnsComplete(allow_failures=1),
            CompletesWithin(120.0),
        ],
        tags=["multi_turn", "chaos_turn_2"],
    ),
    Scenario(
        name="multi-turn-rate-limit-turn2",
        description="Rate limit kicks in on the second turn",
        agent=run_support_agent,
        turns=[
            Turn("Check my order ORD-67890"),
            Turn("Now check the shipping status for that order"),
        ],
        chaos=[llm_rate_limit().on_turn(2).after_calls(1)],
        assertions=[
            TurnCompletes(turn=1),
            AllTurnsComplete(allow_failures=1),
            MaxTotalLLMCalls(15),
        ],
        tags=["multi_turn", "rate_limit"],
    ),
    Scenario(
        name="multi-turn-dynamic-escalation",
        description="Customer escalates if first response is unsatisfactory",
        agent=run_support_agent,
        turns=[
            Turn(
                input="What happened to my order ORD-67890?",
                chaos=[
                    tool_error("Service temporarily unavailable").for_tool(
                        "lookup_order"
                    )
                ],
            ),
            Turn(input=escalate_if_unhappy),  # Dynamic based on turn 1
        ],
        assertions=[AllTurnsComplete(allow_failures=1), CompletesWithin(180.0)],
        tags=["multi_turn", "dynamic", "escalation"],
    ),
    Scenario(
        name="multi-turn-progressive-issues",
        description="3-turn journey with increasing problems",
        agent=run_support_agent,
        turns=[
            Turn("Where is my order ORD-67890?"),
            Turn(
                input="Can I get a refund since it's delayed?",
                chaos=[
                    tool_error("Refund service offline").for_tool(
                        "check_refund_eligibility"
                    )
                ],
            ),
            Turn("Fine, just connect me with a human agent please."),
        ],
        chaos=[llm_rate_limit().on_turn(3).after_calls(1)],
        assertions=[
            TurnCompletes(turn=1),
            AllTurnsComplete(allow_failures=2),
            CompletesWithin(180.0),
        ],
        tags=["multi_turn", "progressive_failure"],
    ),
    Scenario(
        name="multi-turn-frustrated-customer",
        description="Customer with progressively increasing frustration",
        agent=run_support_agent,
        turns=[
            Turn(input=progressive_frustration),
            Turn(input=progressive_frustration),
            Turn(input=progressive_frustration),
        ],
        chaos=[tool_error("Network error").for_tool("lookup_order").on_turn(2)],
        assertions=[AllTurnsComplete(allow_failures=1), CompletesWithin(180.0)],
        tags=["multi_turn", "frustration"],
    ),
    Scenario(
        name="multi-turn-recovery",
        description="Agent fails on turn 2 but recovers on turn 3",
        agent=run_support_agent,
        turns=[
            Turn("Check order ORD-67890"),
            Turn(
                input="Now process a refund for it",
                chaos=[tool_error("System error").for_tool("process_refund")],
            ),
            Turn("Try the refund again please"),
        ],
        assertions=[
            TurnCompletes(turn=1),
            TurnCompletes(turn=2, expect_error=True),
            TurnCompletes(turn=3),
            RecoveredAfterFailure(failed_turn=2),
        ],
        tags=["multi_turn", "recovery"],
    ),
    Scenario(
        name="multi-turn-verified-journey",
        description="Multi-turn with specific content assertions per turn",
        agent=run_support_agent,
        turns=[
            Turn("What's in my order ORD-67890?"),
            Turn("Is the Laptop Stand in stock?"),
        ],
        assertions=[
            AllTurnsComplete(),
            TurnResponseContains(substring="laptop stand", turn=1),
            CompletesWithin(120.0),
        ],
        tags=["multi_turn", "verified"],
    ),
]
