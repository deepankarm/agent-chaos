"""E-commerce Support Agent using pydantic-ai.

A production-style customer support agent for testing with agent-chaos.
Includes realistic tools for order management, refunds, and shipping.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from agent_chaos import ChaosContext
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel


def _days_ago(days: int) -> str:
    """Return a date string for N days ago."""
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _days_ago_time(days: int, hour: int = 10, minute: int = 0) -> str:
    """Return a datetime string for N days ago at a specific time."""
    dt = datetime.now() - timedelta(days=days)
    return dt.replace(hour=hour, minute=minute).strftime("%Y-%m-%d %H:%M")


def get_anthropic_model() -> AnthropicModel:
    return AnthropicModel(model_name="claude-sonnet-4-5-20250929")


# =============================================================================
# Initial State Templates (used to create fresh copies per scenario run)
# =============================================================================


def _get_initial_orders() -> dict[str, dict[str, Any]]:
    """Generate orders with dynamic dates relative to today."""
    return {
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
                {
                    "name": "USB-C Cable",
                    "sku": "USB-C-2M",
                    "quantity": 2,
                    "price": 12.99,
                },
            ],
            "total": 105.97,
            "shipping_address": "123 Main St, Seattle, WA 98101",
            "order_date": _days_ago(10),  # 10 days ago
            "delivery_date": _days_ago(5),  # delivered 5 days ago
        },
        "ORD-67890": {
            "order_id": "ORD-67890",
            "customer_id": "CUST-001",
            "status": "shipped",
            "items": [
                {
                    "name": "Laptop Stand",
                    "sku": "LS-PRO",
                    "quantity": 1,
                    "price": 49.99,
                },
            ],
            "total": 54.99,
            "shipping_address": "123 Main St, Seattle, WA 98101",
            "order_date": _days_ago(3),  # 3 days ago
            "tracking_number": "1Z999AA10123456784",
            "carrier": "UPS",
            "estimated_delivery": _days_ago(-1),  # tomorrow
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
            "order_date": _days_ago(1),  # yesterday
        },
    }


_INITIAL_REFUNDS: dict[str, dict[str, Any]] = {
    "REF-001": {
        "refund_id": "REF-001",
        "order_id": "ORD-12345",
        "amount": 79.99,
        "status": "completed",
        "reason": "Item defective",
        "processed_date": "2024-12-22",
    },
}


def _get_initial_shipping() -> dict[str, list[dict[str, Any]]]:
    """Generate shipping events with dynamic timestamps."""
    return {
        "1Z999AA10123456784": [
            {
                "timestamp": _days_ago_time(3, hour=10),
                "location": "Los Angeles, CA",
                "status": "Package picked up from warehouse",
            },
            {
                "timestamp": _days_ago_time(2, hour=6),
                "location": "San Francisco, CA",
                "status": "In transit",
            },
            {
                "timestamp": _days_ago_time(1, hour=14),
                "location": "Portland, OR",
                "status": "At distribution center",
            },
            {
                "timestamp": _days_ago_time(0, hour=8),
                "location": "Seattle, WA",
                "status": "Out for delivery",
            },
        ],
    }


_INITIAL_PRODUCTS: dict[str, dict[str, Any]] = {
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


# =============================================================================
# Dependencies (with isolated state per scenario run)
# =============================================================================


@dataclass
class SupportDeps:
    """Dependencies for the support agent.

    Each scenario run gets its own instance with fresh copies of the DBs,
    ensuring complete isolation between concurrent/sequential runs.
    """

    customer_id: str = "CUST-001"
    session_id: str = field(
        default_factory=lambda: f"sess-{random.randint(1000, 9999)}"
    )
    escalation_requested: bool = False
    refund_attempts: int = 0

    # Isolated state per run (fresh copies with dynamic dates)
    orders_db: dict[str, dict[str, Any]] = field(default_factory=_get_initial_orders)
    refunds_db: dict[str, dict[str, Any]] = field(
        default_factory=lambda: deepcopy(_INITIAL_REFUNDS)
    )
    shipping_db: dict[str, list[dict[str, Any]]] = field(
        default_factory=_get_initial_shipping
    )
    products_db: dict[str, dict[str, Any]] = field(
        default_factory=lambda: deepcopy(_INITIAL_PRODUCTS)
    )


async def lookup_order(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """Look up an order by its ID.

    Args:
        order_id: The order ID to look up (e.g., ORD-12345)

    Returns:
        JSON string with order details or error message
    """
    order = ctx.deps.orders_db.get(order_id)
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


async def get_shipping_status(
    ctx: RunContext[SupportDeps], tracking_number: str
) -> str:
    """Get real-time shipping status for a tracking number.

    Args:
        tracking_number: The carrier tracking number

    Returns:
        JSON string with shipping updates
    """
    events = ctx.deps.shipping_db.get(tracking_number)
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


async def check_refund_eligibility(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """Check if an order is eligible for refund.

    Args:
        order_id: The order ID to check

    Returns:
        JSON string with eligibility details
    """
    order = ctx.deps.orders_db.get(order_id)
    if not order:
        return json.dumps({"eligible": False, "reason": "Order not found"})

    # Check if already refunded
    for refund in ctx.deps.refunds_db.values():
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


async def process_refund(
    ctx: RunContext[SupportDeps], order_id: str, reason: str
) -> str:
    """Process a refund for an order.

    Args:
        order_id: The order ID to refund
        reason: The reason for the refund

    Returns:
        JSON string with refund confirmation or error
    """
    order = ctx.deps.orders_db.get(order_id)
    if not order:
        return json.dumps({"success": False, "error": "Order not found"})

    # Check eligibility first
    for refund in ctx.deps.refunds_db.values():
        if refund["order_id"] == order_id:
            return json.dumps(
                {
                    "success": False,
                    "error": "Order already refunded",
                }
            )

    # Create refund (mutates isolated deps, not global state)
    refund_id = f"REF-{random.randint(100, 999)}"
    refund = {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": order["total"],
        "status": "pending",
        "reason": reason,
        "processed_date": datetime.now().strftime("%Y-%m-%d"),
    }
    ctx.deps.refunds_db[refund_id] = refund

    return json.dumps(
        {
            "success": True,
            "refund_id": refund_id,
            "amount": order["total"],
            "status": "pending",
            "estimated_processing": "3-5 business days",
        }
    )


async def get_refund_status(ctx: RunContext[SupportDeps], refund_id: str) -> str:
    """Get the status of a refund.

    Args:
        refund_id: The refund ID to check

    Returns:
        JSON string with refund status
    """
    refund = ctx.deps.refunds_db.get(refund_id)
    if not refund:
        return json.dumps({"error": f"Refund {refund_id} not found"})

    return json.dumps(refund)


async def escalate_to_human(
    ctx: RunContext[SupportDeps], issue_summary: str, priority: str = "normal"
) -> str:
    """Escalate the issue to a human support agent.

    Args:
        issue_summary: Brief summary of the customer's issue
        priority: Priority level (low, normal, high, urgent)

    Returns:
        JSON string with escalation confirmation
    """
    ticket_id = f"TKT-{random.randint(10000, 99999)}"

    # Mark escalation in deps
    ctx.deps.escalation_requested = True

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


async def check_product_availability(ctx: RunContext[SupportDeps], sku: str) -> str:
    """Check product availability and stock levels.

    Args:
        sku: The product SKU to check

    Returns:
        JSON string with availability info
    """
    product = ctx.deps.products_db.get(sku)
    if not product:
        return json.dumps({"error": f"Product {sku} not found"})

    return json.dumps(product)


def get_tools() -> list[Callable]:
    return [
        lookup_order,
        get_shipping_status,
        check_refund_eligibility,
        process_refund,
        get_refund_status,
    ]


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

    for tool in get_tools():
        agent.tool(tool)

    return agent


# =============================================================================
# Entry Point for Chaos Scenarios
# =============================================================================


def build_message_history(ctx: ChaosContext):
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    message_history = []
    for msg in ctx.get_message_history():
        if msg["role"] == "user":
            message_history.append(
                ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            )
        elif msg["role"] == "assistant":
            message_history.append(
                ModelResponse(parts=[TextPart(content=msg["content"])])
            )

    return message_history


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

    # Get or create deps - reuse across turns within the same scenario run
    deps = ctx.agent_state.get("deps")
    if deps is None:
        deps = SupportDeps()
        ctx.agent_state["deps"] = deps

    # Run with history for multi-turn conversation continuity
    history = build_message_history(ctx) or None
    result = await agent.run(query, deps=deps, message_history=history)
    return str(result.output)
