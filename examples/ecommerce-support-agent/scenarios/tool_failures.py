from __future__ import annotations

import json

from agent_chaos import ChaosContext, Turn
from agent_chaos.chaos import tool_empty, tool_error, tool_mutate, tool_timeout
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    MaxTotalLLMCalls,
    MinChaosInjected,
    Scenario,
)

from agent import run_support_agent

# =============================================================================
# Tool Mutators (for tool_mutate chaos)
# =============================================================================


def corrupt_order_status(tool_name: str, result: str) -> str:
    """Corrupt order status to test agent response to stale/wrong data."""
    if "lookup_order" in tool_name:
        try:
            data = json.loads(result)
            if "status" in data:
                data["status"] = "returned_to_sender"
                data["note"] = "Package undeliverable - address invalid"
            return json.dumps(data)
        except json.JSONDecodeError:
            pass
    return result


def corrupt_refund_amount(ctx: ChaosContext, tool_name: str, result: str) -> str:
    """Corrupt refund amount to test agent handling of data integrity issues."""
    if "check_refund_eligibility" in tool_name or "process_refund" in tool_name:
        try:
            data = json.loads(result)
            if "refundable_amount" in data:
                data["refundable_amount"] = data["refundable_amount"] * 10
            if "amount" in data:
                data["amount"] = -50.00
            return json.dumps(data)
        except json.JSONDecodeError:
            pass
    return result


def inject_shipping_delay(tool_name: str, result: str) -> str:
    """Inject unexpected shipping delays into tracking data."""
    if "get_shipping_status" in tool_name:
        try:
            data = json.loads(result)
            if "events" in data:
                data["events"].append(
                    {
                        "timestamp": "2024-12-26 09:00",
                        "location": "Unknown",
                        "status": "EXCEPTION: Package held at customs - documentation required",
                    }
                )
                data["current_status"] = "EXCEPTION: Customs hold"
            return json.dumps(data)
        except json.JSONDecodeError:
            pass
    return result


# =============================================================================
# Tool Failure Scenarios
# =============================================================================

tool_failure_scenarios = [
    Scenario(
        name="tool-error-order-lookup",
        description="Order lookup tool returns database error - agent should apologize and offer alternatives",
        agent=run_support_agent,
        turns=[
            Turn(
                input="What's the status of my order ORD-67890?",
                chaos=[
                    tool_error("Database connection timeout").for_tool("lookup_order")
                ],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "lookup_order", "error"],
    ),
    Scenario(
        name="tool-error-shipping-unavailable",
        description="Shipping API returns service unavailable - agent should inform customer",
        agent=run_support_agent,
        turns=[
            Turn(
                input="Where is my package? Tracking: 1Z999AA10123456784",
                chaos=[
                    tool_error("Carrier API service unavailable").for_tool(
                        "get_shipping_status"
                    )
                ],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "get_shipping_status", "error"],
    ),
    Scenario(
        name="tool-timeout-refund-processing",
        description="Refund processing times out - agent should not confirm refund was processed",
        agent=run_support_agent,
        turns=[
            Turn(
                input="I want a refund for order ORD-67890. The product is defective.",
                chaos=[tool_timeout(timeout_seconds=30.0).for_tool("process_refund")],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(90.0)],
        tags=["tool_failure", "process_refund", "timeout"],
    ),
    Scenario(
        name="tool-empty-order-lookup",
        description="Order lookup returns empty response - agent should handle gracefully",
        agent=run_support_agent,
        turns=[
            Turn(
                input="Can you find order ORD-12345?",
                chaos=[tool_empty().for_tool("lookup_order")],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "lookup_order", "empty"],
    ),
    Scenario(
        name="tool-mutate-stale-order-status",
        description="Order status is corrupted/stale - tests if agent notices inconsistencies",
        agent=run_support_agent,
        turns=[
            Turn(
                input="What happened to my order ORD-67890? It was supposed to arrive yesterday.",
                chaos=[tool_mutate(corrupt_order_status).for_tool("lookup_order")],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "lookup_order", "mutation"],
    ),
    Scenario(
        name="tool-mutate-wrong-refund-amount",
        description="Refund amount is incorrect - agent should not process suspicious amounts",
        agent=run_support_agent,
        turns=[
            Turn(
                input="Please process a refund for my order ORD-67890",
                chaos=[tool_mutate(corrupt_refund_amount)],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "data_corruption"],
    ),
    Scenario(
        name="tool-mutate-shipping-exception",
        description="Shipping tracking shows unexpected customs hold - agent should explain the situation",
        agent=run_support_agent,
        turns=[
            Turn(
                input="My package tracking number is 1Z999AA10123456784. When will it arrive?",
                chaos=[tool_mutate(inject_shipping_delay)],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(60.0), MinChaosInjected(1)],
        tags=["tool_failure", "shipping_exception"],
    ),
    Scenario(
        name="tool-error-cascade-failure",
        description="Multiple tools fail at once - agent should escalate to human support",
        agent=run_support_agent,
        turns=[
            Turn(
                input="I need to check my order ORD-67890 and get a refund. The package never arrived!",
                chaos=[
                    tool_error("Service unavailable").for_tool("lookup_order"),
                    tool_error("Service unavailable").for_tool(
                        "check_refund_eligibility"
                    ),
                    tool_error("Service unavailable").for_tool("process_refund"),
                ],
            ),
        ],
        assertions=[AllTurnsComplete(), CompletesWithin(90.0), MaxTotalLLMCalls(10)],
        tags=["tool_failure", "cascade"],
    ),
]
