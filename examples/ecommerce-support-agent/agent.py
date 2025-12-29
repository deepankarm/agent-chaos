"""E-commerce Support Agent using pydantic-ai.

A production-style customer support agent for testing with agent-chaos.
Includes realistic tools for order management, refunds, and shipping.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agent_chaos import ChaosContext
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel


def get_anthropic_model() -> AnthropicModel:
    return AnthropicModel(model_name="claude-sonnet-4-5-20250929")


# =============================================================================
# Mock Data Store (simulates database)
# =============================================================================

ORDERS_DB: dict[str, dict[str, Any]] = {
    "ORD-12345": {
        "order_id": "ORD-12345",
        "customer_id": "CUST-001",
        "status": "delivered",
        "items": [
            {
                "name": "Wireless Headphones",
                "sku": "WH-100",
                "quantity": 1,
                "price": 79.99,
            },
            {"name": "USB-C Cable", "sku": "USB-C-2M", "quantity": 2, "price": 12.99},
        ],
        "total": 105.97,
        "shipping_address": "123 Main St, Seattle, WA 98101",
        "order_date": "2024-12-15",
        "delivery_date": "2024-12-20",
    },
    "ORD-67890": {
        "order_id": "ORD-67890",
        "customer_id": "CUST-001",
        "status": "shipped",
        "items": [
            {"name": "Laptop Stand", "sku": "LS-PRO", "quantity": 1, "price": 49.99},
        ],
        "total": 54.99,
        "shipping_address": "123 Main St, Seattle, WA 98101",
        "order_date": "2024-12-22",
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS",
        "estimated_delivery": "2024-12-28",
    },
    "ORD-11111": {
        "order_id": "ORD-11111",
        "customer_id": "CUST-002",
        "status": "processing",
        "items": [
            {
                "name": "Mechanical Keyboard",
                "sku": "KB-MX",
                "quantity": 1,
                "price": 129.99,
            },
        ],
        "total": 129.99,
        "shipping_address": "456 Oak Ave, Portland, OR 97201",
        "order_date": "2024-12-26",
    },
}

REFUNDS_DB: dict[str, dict[str, Any]] = {
    "REF-001": {
        "refund_id": "REF-001",
        "order_id": "ORD-12345",
        "amount": 79.99,
        "status": "completed",
        "reason": "Item defective",
        "processed_date": "2024-12-22",
    },
}

SHIPPING_TRACKING: dict[str, list[dict[str, Any]]] = {
    "1Z999AA10123456784": [
        {
            "timestamp": "2024-12-22 10:00",
            "location": "Seattle, WA",
            "status": "Package picked up",
        },
        {
            "timestamp": "2024-12-23 06:00",
            "location": "Portland, OR",
            "status": "In transit",
        },
        {
            "timestamp": "2024-12-24 08:00",
            "location": "San Francisco, CA",
            "status": "At distribution center",
        },
        {
            "timestamp": "2024-12-25 14:00",
            "location": "Los Angeles, CA",
            "status": "Out for delivery",
        },
    ],
}


@dataclass
class SupportDeps:
    """Dependencies for the support agent."""

    customer_id: str = "CUST-001"
    session_id: str = field(
        default_factory=lambda: f"sess-{random.randint(1000, 9999)}"
    )
    escalation_requested: bool = False
    refund_attempts: int = 0


async def lookup_order(ctx, order_id: str) -> str:
    """Look up an order by its ID.

    Args:
        order_id: The order ID to look up (e.g., ORD-12345)

    Returns:
        JSON string with order details or error message
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return json.dumps({"error": f"Order {order_id} not found"})

    return json.dumps(
        {
            "order_id": order["order_id"],
            "status": order["status"],
            "items": order["items"],
            "total": order["total"],
            "order_date": order["order_date"],
            "shipping_address": order["shipping_address"],
            "delivery_date": order.get("delivery_date"),
            "tracking_number": order.get("tracking_number"),
            "carrier": order.get("carrier"),
        }
    )


async def get_shipping_status(ctx, tracking_number: str) -> str:
    """Get real-time shipping status for a tracking number.

    Args:
        tracking_number: The carrier tracking number

    Returns:
        JSON string with shipping updates
    """
    events = SHIPPING_TRACKING.get(tracking_number)
    if not events:
        return json.dumps({"error": f"No tracking info for {tracking_number}"})

    return json.dumps(
        {
            "tracking_number": tracking_number,
            "events": events,
            "current_status": events[-1]["status"],
            "current_location": events[-1]["location"],
        }
    )


async def check_refund_eligibility(ctx, order_id: str) -> str:
    """Check if an order is eligible for refund.

    Args:
        order_id: The order ID to check

    Returns:
        JSON string with eligibility details
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return json.dumps({"eligible": False, "reason": "Order not found"})

    # Check if already refunded
    for refund in REFUNDS_DB.values():
        if refund["order_id"] == order_id:
            return json.dumps(
                {
                    "eligible": False,
                    "reason": f"Order already refunded (Refund ID: {refund['refund_id']})",
                }
            )

    # Check if within refund window (30 days)
    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
    if datetime.now() - order_date > timedelta(days=30):
        return json.dumps(
            {
                "eligible": False,
                "reason": "Order is outside 30-day refund window",
            }
        )

    return json.dumps(
        {
            "eligible": True,
            "order_id": order_id,
            "refundable_amount": order["total"],
            "items": [item["name"] for item in order["items"]],
        }
    )


async def process_refund(ctx, order_id: str, reason: str) -> str:
    """Process a refund for an order.

    Args:
        order_id: The order ID to refund
        reason: The reason for the refund

    Returns:
        JSON string with refund confirmation or error
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return json.dumps({"success": False, "error": "Order not found"})

    # Check eligibility first
    for refund in REFUNDS_DB.values():
        if refund["order_id"] == order_id:
            return json.dumps(
                {
                    "success": False,
                    "error": "Order already refunded",
                }
            )

    # Create refund
    refund_id = f"REF-{random.randint(100, 999)}"
    refund = {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": order["total"],
        "status": "pending",
        "reason": reason,
        "processed_date": datetime.now().strftime("%Y-%m-%d"),
    }
    REFUNDS_DB[refund_id] = refund

    return json.dumps(
        {
            "success": True,
            "refund_id": refund_id,
            "amount": order["total"],
            "status": "pending",
            "estimated_processing": "3-5 business days",
        }
    )


async def get_refund_status(ctx, refund_id: str) -> str:
    """Get the status of a refund.

    Args:
        refund_id: The refund ID to check

    Returns:
        JSON string with refund status
    """
    refund = REFUNDS_DB.get(refund_id)
    if not refund:
        return json.dumps({"error": f"Refund {refund_id} not found"})

    return json.dumps(refund)


async def escalate_to_human(ctx, issue_summary: str, priority: str = "normal") -> str:
    """Escalate the issue to a human support agent.

    Args:
        issue_summary: Brief summary of the customer's issue
        priority: Priority level (low, normal, high, urgent)

    Returns:
        JSON string with escalation confirmation
    """
    ticket_id = f"TKT-{random.randint(10000, 99999)}"

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "priority": priority,
            "estimated_response": (
                "2-4 hours" if priority in ["high", "urgent"] else "24 hours"
            ),
            "message": f"Your issue has been escalated. A human agent will contact you soon. Reference: {ticket_id}",
        }
    )


async def check_product_availability(ctx, sku: str) -> str:
    """Check product availability and stock levels.

    Args:
        sku: The product SKU to check

    Returns:
        JSON string with availability info
    """
    # Simulated product catalog
    products = {
        "WH-100": {"name": "Wireless Headphones", "in_stock": True, "quantity": 150},
        "USB-C-2M": {"name": "USB-C Cable", "in_stock": True, "quantity": 500},
        "LS-PRO": {
            "name": "Laptop Stand",
            "in_stock": False,
            "quantity": 0,
            "restock_date": "2025-01-15",
        },
        "KB-MX": {"name": "Mechanical Keyboard", "in_stock": True, "quantity": 25},
    }

    product = products.get(sku)
    if not product:
        return json.dumps({"error": f"Product {sku} not found"})

    return json.dumps(product)


def create_support_agent() -> Agent[SupportDeps, str]:
    """Create the e-commerce support agent with all tools."""
    agent = Agent[SupportDeps, str](
        model=get_anthropic_model(),
        deps_type=SupportDeps,
        system_prompt="""You are a helpful e-commerce customer support agent for TechMart, an online electronics retailer.

Your responsibilities:
1. Help customers track their orders and shipments
2. Process refunds when eligible and requested
3. Answer questions about products and availability
4. Escalate complex issues to human agents when needed

Guidelines:
- Always verify order details before taking actions
- Be empathetic and professional
- If a tool returns an error, explain the issue clearly to the customer
- For refunds, always check eligibility first
- Escalate to human support if the customer is frustrated or the issue is complex
- Keep responses concise but helpful

You have access to tools for order lookup, shipping tracking, refunds, and escalation.""",
    )

    agent.tool(lookup_order)
    agent.tool(get_shipping_status)
    agent.tool(check_refund_eligibility)
    agent.tool(process_refund)
    agent.tool(get_refund_status)
    agent.tool(escalate_to_human)
    agent.tool(check_product_availability)

    return agent


async def run_support_agent(ctx: ChaosContext, query: str) -> str:
    """Run the support agent with a customer query.

    This is the main entry point for chaos scenarios.

    Args:
        ctx: The chaos context (injected by agent-chaos)
        query: The customer's query/message

    Returns:
        The agent's response as a string
    """
    agent = create_support_agent()
    deps = SupportDeps()

    # Get message history from previous turns (if any)
    message_history = ctx.agent_state.get("message_history")

    # Run with history for multi-turn conversation continuity
    result = await agent.run(query, deps=deps, message_history=message_history)

    # Store message history for next turn
    ctx.agent_state["message_history"] = result.all_messages()

    return str(result.output)
