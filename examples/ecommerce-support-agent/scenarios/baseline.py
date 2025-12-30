"""Baseline scenarios - happy path tests without chaos.

These establish the expected behavior for comparison with chaos scenarios.
All scenarios use the Turn-based architecture.
"""

from __future__ import annotations

from agent_chaos import Turn
from agent_chaos.scenario import AllTurnsComplete, CompletesWithin, MaxTotalLLMCalls, Scenario

from agent import run_support_agent

baseline_scenarios = [
    # Order lookup - happy path
    Scenario(
        name="baseline-order-lookup",
        description="Customer checks order status - no chaos",
        agent=run_support_agent,
        turns=[
            Turn("What's the status of my order ORD-67890?"),
        ],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(60.0),
            MaxTotalLLMCalls(5),
        ],
        tags=["baseline", "order_lookup"],
    ),
    # Shipping tracking - happy path
    Scenario(
        name="baseline-shipping-tracking",
        description="Customer tracks shipment - no chaos",
        agent=run_support_agent,
        turns=[
            Turn("Can you track my package? The tracking number is 1Z999AA10123456784"),
        ],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(60.0),
            MaxTotalLLMCalls(5),
        ],
        tags=["baseline", "shipping_tracking"],
    ),
    # Refund inquiry - happy path
    Scenario(
        name="baseline-refund-inquiry",
        description="Customer asks about refund eligibility - no chaos",
        agent=run_support_agent,
        turns=[
            Turn("Can I get a refund for order ORD-67890? The laptop stand doesn't fit my desk."),
        ],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(60.0),
            MaxTotalLLMCalls(6),
        ],
        tags=["baseline", "refund_inquiry"],
    ),
    # Product availability - happy path
    Scenario(
        name="baseline-product-availability",
        description="Customer checks if product is in stock - no chaos",
        agent=run_support_agent,
        turns=[
            Turn("Is the Mechanical Keyboard (SKU: KB-MX) still available?"),
        ],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(60.0),
            MaxTotalLLMCalls(4),
        ],
        tags=["baseline", "product_availability"],
    ),
]
