"""DeepEval integration scenarios for semantic quality evaluation under chaos.

These scenarios demonstrate using LLM-as-judge evaluation to verify that
agents produce correct, helpful responses even when chaos is injected.

This is the real power of agent-chaos: not just "did it complete?" but
"did it complete CORRECTLY under adversity?"

Requirements:
    pip install deepeval  # Or: uv add --dev deepeval
    Set ANTHROPIC_API_KEY environment variable for LLM-as-judge evaluation.
"""

from __future__ import annotations

from agent_chaos import Turn, TurnResult
from agent_chaos.chaos import tool_error, tool_mutate

# DeepEval imports - lazy loaded in the integration
from agent_chaos.integrations.deepeval import as_assertion
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    MaxTotalLLMCalls,
    Scenario,
    TurnCompletes,
)

from agent import run_support_agent


def get_eval_model():
    """Get or create the shared Anthropic evaluation model.

    Uses DeepEval's native AnthropicModel - requires ANTHROPIC_API_KEY env var.
    """
    from deepeval.models import AnthropicModel

    return AnthropicModel(model="claude-sonnet-4-20250514", temperature=0)


# =============================================================================
# Custom Tool Mutators (Semantic Chaos)
# =============================================================================


def corrupt_order_status(tool_name: str, result: str) -> str:
    """Corrupt order status to test if agent catches implausible data."""
    if "shipped" in result.lower():
        # Change shipped to delivered with wrong dates
        return result.replace("shipped", "delivered").replace(
            "estimated_delivery", "was_delivered_on"
        )
    return result


def inject_wrong_refund_amount(tool_name: str, result: str) -> str:
    """Return wrong refund amount to test if agent validates data."""
    import json

    try:
        data = json.loads(result)
        if "refundable_amount" in data:
            # 10x the actual amount - agent should catch this
            data["refundable_amount"] = data["refundable_amount"] * 10
            data["note"] = "Special VIP discount applied"
        return json.dumps(data)
    except json.JSONDecodeError:
        return result


def hallucinate_product_info(tool_name: str, result: str) -> str:
    """Replace product info with plausible but wrong data."""
    import json

    try:
        data = json.loads(result)
        if "name" in data:
            data["name"] = "Premium " + data["name"] + " Pro Max"
            data["in_stock"] = True
            data["quantity"] = 9999
        return json.dumps(data)
    except json.JSONDecodeError:
        return result


# =============================================================================
# Dynamic Input Generators
# =============================================================================


def verify_previous_info(history: list[TurnResult]) -> str:
    """Follow up asking for confirmation of previously provided info."""
    if not history:
        return "Can you check my order status?"

    last = history[-1]
    if "refund" in last.response.lower():
        return "Wait, that refund amount seems very high. Can you double-check it?"
    elif "delivered" in last.response.lower():
        return "That doesn't match what I see in my delivery tracking. Are you sure?"
    return "Thanks, that helps."


# =============================================================================
# DeepEval Metrics Setup
# =============================================================================


def get_task_completion_metric():
    """Create a GEval metric for task completion assessment."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="task-completion",
        criteria="""Evaluate whether the customer support agent successfully addressed
        the customer's request. Consider:
        1. Did the agent understand what the customer wanted?
        2. Did the agent take appropriate actions (lookup order, check refund, etc)?
        3. Did the agent provide a clear, helpful response?
        4. If there were errors, did the agent handle them gracefully?""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
        model=get_eval_model(),
    )


def get_data_accuracy_metric():
    """Create a GEval metric for checking if agent doesn't propagate bad data."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="data-accuracy",
        criteria="""Evaluate whether the agent's response contains accurate,
        plausible information. Check for:
        1. Prices and amounts that seem reasonable
        2. Dates and times that make sense
        3. Product information that matches what a real e-commerce site would have
        4. Any signs the agent is repeating clearly wrong data from tools

        The agent should NOT blindly repeat implausible data like:
        - Refund amounts 10x the order value
        - Stock quantities of 9999
        - Delivery dates in the past for items not yet shipped""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        model=get_eval_model(),
    )


def get_error_handling_metric():
    """Create a GEval metric for graceful error handling."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="error-handling",
        criteria="""Evaluate how well the agent handled errors or failures.
        A good response should:
        1. Acknowledge when something went wrong
        2. Not pretend to have information it couldn't retrieve
        3. Offer alternatives or next steps
        4. Maintain a helpful, professional tone
        5. Not expose internal error messages to the customer""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
        model=get_eval_model(),
    )


def get_tool_correctness_metric():
    """Create a ToolCorrectnessMetric for checking tool usage."""
    from deepeval.metrics import ToolCorrectnessMetric

    return ToolCorrectnessMetric(
        threshold=0.5,
        include_reason=True,
        model=get_eval_model(),
    )


deepeval_scenarios = [
    Scenario(
        name="deepeval-baseline-task-completion",
        description="Verify basic task completion with LLM-as-judge (no chaos)",
        agent=run_support_agent,
        turns=[
            Turn("What's the status of my order ORD-67890?"),
            Turn("Great, when will it arrive?"),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(120.0),
            as_assertion(get_task_completion_metric),
        ],
        tags=["deepeval", "baseline"],
    ),
    Scenario(
        name="deepeval-error-handling",
        description="Verify graceful error handling when tools fail",
        agent=run_support_agent,
        turns=[
            Turn(
                input="Check my order ORD-67890",
                chaos=[
                    tool_error("Order service temporarily unavailable").for_tool(
                        "lookup_order"
                    )
                ],
            ),
            Turn("Can you try again or help me another way?"),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(allow_failures=1),
            CompletesWithin(120.0),
            as_assertion(get_error_handling_metric),
        ],
        tags=["deepeval", "error_handling"],
    ),
    Scenario(
        name="deepeval-data-corruption-detection",
        description="Test if agent catches and handles corrupted tool results",
        agent=run_support_agent,
        turns=[
            Turn(
                input="How much refund can I get for order ORD-67890?",
                chaos=[
                    tool_mutate(inject_wrong_refund_amount).for_tool(
                        "check_refund_eligibility"
                    )
                ],
            ),
            Turn(input=verify_previous_info),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(120.0),
            as_assertion(
                get_data_accuracy_metric,
                name="detects-wrong-refund-amount",
            ),
        ],
        tags=["deepeval", "data_corruption"],
    ),
    Scenario(
        name="deepeval-tool-usage-verification",
        description="Verify agent uses correct tools in multi-turn conversation",
        agent=run_support_agent,
        turns=[
            Turn("What products are available with SKU KB-MX?"),
            Turn("Great, can you check if order ORD-67890 is eligible for a refund?"),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(120.0),
            as_assertion(
                get_tool_correctness_metric,
                expected_tools=[
                    "check_product_availability",
                    "check_refund_eligibility",
                ],
            ),
            as_assertion(get_task_completion_metric),
        ],
        tags=["deepeval", "tool_verification"],
    ),
    Scenario(
        name="deepeval-multi-turn-chaos-quality",
        description="Full multi-turn with chaos injection AND LLM quality evaluation",
        agent=run_support_agent,
        turns=[
            Turn("I need help with order ORD-67890"),
            Turn(
                input="Can I get a refund? The item was defective.",
                chaos=[
                    tool_error("Connection timeout").for_tool(
                        "check_refund_eligibility"
                    )
                ],
            ),
            Turn("Please try again, this is urgent!"),
        ],
        chaos=[],
        assertions=[
            TurnCompletes(turn=1),
            AllTurnsComplete(allow_failures=1),
            MaxTotalLLMCalls(15),
            CompletesWithin(180.0),
            as_assertion(get_task_completion_metric),
            as_assertion(get_error_handling_metric),
        ],
        tags=["deepeval", "comprehensive"],
    ),
    Scenario(
        name="deepeval-hallucination-detection",
        description="Test if agent catches hallucinated product information",
        agent=run_support_agent,
        turns=[
            Turn(
                input="Is the laptop stand with SKU LS-PRO available?",
                chaos=[
                    tool_mutate(hallucinate_product_info).for_tool(
                        "check_product_availability"
                    )
                ],
            ),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(),
            as_assertion(
                get_data_accuracy_metric,
                name="detects-hallucinated-product",
            ),
        ],
        tags=["deepeval", "hallucination"],
    ),
    Scenario(
        name="deepeval-per-turn-and-conversation-evaluation",
        description="Evaluate each turn separately using Turn.assertions and evaluate the entire conversation holistically",
        agent=run_support_agent,
        turns=[
            Turn(
                input="What's the status of order ORD-12345?",
                assertions=[
                    as_assertion(get_task_completion_metric, name="turn1-quality")
                ],
            ),
            Turn(
                input="Can you tell me when it was delivered?",
                assertions=[
                    as_assertion(get_task_completion_metric, name="turn2-quality")
                ],
            ),
            Turn(
                input="I'd like to check if order ORD-67890 is eligible for a refund.",
                assertions=[
                    as_assertion(get_task_completion_metric, name="turn3-quality")
                ],
            ),
            Turn(
                input="What about order ORD-11111? What's its status?",
                assertions=[
                    as_assertion(get_task_completion_metric, name="turn4-quality")
                ],
            ),
        ],
        chaos=[],
        assertions=[
            AllTurnsComplete(),
            CompletesWithin(180.0),
            as_assertion(
                get_task_completion_metric,
                name="conversation-task-completion",
            ),
        ],
        tags=["deepeval", "per_turn_evaluation"],
    ),
]
