"""Fuzz scenarios — random chaos generation for exploratory testing.

These scenarios use fuzz_chaos() to generate random chaos configurations,
exploring the failure space to find unexpected agent behaviors.

Run with: agent-chaos run scenarios/fuzz_scenarios.py
Filter fuzz only: agent-chaos run scenarios/ --tag fuzz
"""

from __future__ import annotations

import json

import anthropic
from agent_chaos import (
    ChaosSpace,
    LLMFuzzConfig,
    StreamFuzzConfig,
    ToolFuzzConfig,
    Turn,
    TurnResult,
    fuzz_chaos,
)
from agent_chaos.integrations.deepeval import as_assertion
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    MaxTotalLLMCalls,
    Scenario,
)

from agent import get_tools, run_support_agent


def _get_eval_model():
    """Shared evaluation model for all metrics."""
    from deepeval.models import AnthropicModel

    return AnthropicModel(model="claude-sonnet-4-20250514", temperature=0)


def get_error_handling_metric():
    """Health check: Agent handles errors gracefully without crashing or exposing internals."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="error-handling",
        criteria="""Evaluate how well the agent handled errors or unexpected situations.
        A good response should:
        1. Acknowledge when something went wrong (if applicable)
        2. Not pretend to have information it couldn't retrieve
        3. Offer alternatives or next steps when possible
        4. Maintain a helpful, professional tone
        5. Not expose internal error messages or stack traces to the customer""",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,  # Lower threshold - we expect some degradation under chaos
        model=_get_eval_model(),
    )


def get_response_coherence_metric():
    """Health check: Agent responses are coherent and not gibberish."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="response-coherence",
        criteria="""Evaluate whether the agent's response is coherent and understandable.
        Check that:
        1. The response is in proper English (or appropriate language)
        2. Sentences are grammatically correct and make sense
        3. The response addresses the customer (not random text)
        4. There's no garbled, truncated, or nonsensical output

        This is a basic sanity check - even if the agent couldn't help,
        it should at least respond coherently.""",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=_get_eval_model(),
    )


_anthropic_client = anthropic.Anthropic()


def generate_user_input(
    context: str,
    history: list[TurnResult] | None = None,
    persona: str = "frustrated customer",
) -> str:
    """Use Claude to generate realistic user input based on context.

    Args:
        context: Description of what kind of input to generate
        history: Previous turn results for context
        persona: The persona to adopt (e.g., "frustrated customer", "confused customer")

    Returns:
        Generated user input string
    """
    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for i, turn in enumerate(history, 1):
            history_text += f"Turn {i} - User: {turn.input}\n"
            history_text += f"Turn {i} - Agent: {turn.response[:200]}...\n\n"

    prompt = f"""You are simulating a {persona} contacting e-commerce support.

{context}
{history_text}
Generate a single, realistic customer message. Be natural and varied - don't be generic.
Keep it under 2 sentences. Just output the customer message, nothing else."""

    response = _anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


SUPPORTED_TOOLS = [tool.__name__ for tool in get_tools()]


def corrupt_order_status(tool_name: str, result: str) -> str:
    """Corrupt order status to a contradictory value."""
    if "lookup_order" not in tool_name:
        return result
    try:
        data = json.loads(result)
        if "status" in data:
            # Change to a contradictory status
            statuses = ["cancelled", "lost", "returned_to_sender", "pending_review"]
            data["status"] = statuses[hash(result) % len(statuses)]
        return json.dumps(data)
    except (json.JSONDecodeError, KeyError):
        return result


def corrupt_refund_amount(tool_name: str, result: str) -> str:
    """Corrupt refund amount to test agent's sanity checking."""
    if "refund" not in tool_name:
        return result
    try:
        data = json.loads(result)
        if "amount" in data:
            data["amount"] = data["amount"] * 100  # 100x the amount
        if "refundable_amount" in data:
            data["refundable_amount"] = -50.0  # Negative amount
        return json.dumps(data)
    except (json.JSONDecodeError, KeyError):
        return result


def inject_tracking_anomaly(tool_name: str, result: str) -> str:
    """Inject anomalous tracking data."""
    if "shipping" not in tool_name:
        return result
    try:
        data = json.loads(result)
        if "events" in data:
            data["events"].append(
                {
                    "timestamp": "2024-12-26 00:00",
                    "location": "UNKNOWN",
                    "status": "Package held by customs - contraband detected",
                }
            )
            data["current_status"] = "Package held by customs - contraband detected"
        return json.dumps(data)
    except (json.JSONDecodeError, KeyError):
        return result


def return_malformed_json(_tool_name: str, result: str) -> str:
    """Return malformed JSON to test parsing resilience."""
    return result[: len(result) // 2] + "...truncated..."


def inject_conflicting_data(_tool_name: str, result: str) -> str:
    """Inject data that conflicts with other tool results."""
    try:
        data = json.loads(result)
        # Add conflicting timestamps
        if "order_date" in data:
            data["order_date"] = "2099-01-01"  # Future date
        if "estimated_delivery" in data:
            data["estimated_delivery"] = "2020-01-01"  # Past date
        return json.dumps(data)
    except (json.JSONDecodeError, KeyError):
        return result


# =============================================================================
# Dynamic input generators for complex scenarios (LLM-powered)
# =============================================================================

# Frustration contexts for progressive escalation
_FRUSTRATION_CONTEXTS = [
    "You just realized your order ORD-67890 hasn't arrived yet. Express initial concern.",
    "The agent's response wasn't helpful. You're getting impatient. Demand answers about your order.",
    "You've been waiting too long. You're angry now. Demand a refund or compensation.",
    "Nothing is working. You want to know exactly where your package is AND you're considering returning it.",
    "You've had enough. Demand to speak to a human manager immediately. You're done with the bot.",
]


def frustrated_followup(history: list[TurnResult]) -> str:
    """Generate progressively frustrated follow-up using LLM."""
    turn_num = len(history)
    context_idx = min(turn_num, len(_FRUSTRATION_CONTEXTS) - 1)
    context = _FRUSTRATION_CONTEXTS[context_idx]

    # Add order context
    context += "\nRelevant order ID: ORD-67890 (a Laptop Stand order that's been shipped but delayed)"

    return generate_user_input(
        context=context,
        history=history,
        persona="increasingly frustrated customer",
    )


def adaptive_query(history: list[TurnResult]) -> str:
    """Generate queries based on previous responses using LLM."""
    if not history:
        return generate_user_input(
            context="You want to check the status of your order ORD-67890. This is your first message.",
            persona="concerned customer",
        )

    last_response = history[-1].response.lower()

    # Determine context based on what the agent said
    if "shipped" in last_response or "transit" in last_response:
        context = "The agent mentioned your order is shipped/in transit. Ask for detailed tracking info."
    elif "delivered" in last_response:
        context = "The agent says it was delivered but you never got it! Express frustration and ask about refund."
    elif "refund" in last_response and "eligible" in last_response:
        context = "Good news - you're eligible for a refund. Confirm you want to proceed with the refund."
    elif "escalat" in last_response or "ticket" in last_response:
        context = "Your issue was escalated. Ask when you'll hear back - you need this resolved urgently."
    elif "error" in last_response or "unavailable" in last_response:
        context = "Something went wrong on their end. Ask them to try again."
    else:
        context = "The response was vague. Ask about product availability for the Laptop Stand (LS-PRO) as an alternative."

    return generate_user_input(
        context=context,
        history=history,
        persona="attentive customer",
    )


# =============================================================================
# Complex 5-turn baseline scenario
# =============================================================================

# This baseline exercises all major tools and provides opportunities for
# chaos injection at every turn. The conversation flow:
# 1. Order lookup
# 2. Shipping status check
# 3. Refund eligibility check
# 4. Refund processing or product availability check
# 5. Escalation to human

# Per-turn health check - each response should be coherent
_turn_health_check = [
    as_assertion(get_response_coherence_metric, name="turn-coherence")
]

complex_baseline = Scenario(
    name="baseline-complex",
    description="Complex 5-turn journey covering order, shipping, refund, and escalation",
    agent=run_support_agent,
    turns=[
        Turn(
            "Hi, I need to check on my order ORD-67890. What's the current status?",
            assertions=_turn_health_check,
        ),
        Turn(
            "Thanks. Can you give me the shipping details? "
            "I want to see the tracking history for that order.",
            assertions=_turn_health_check,
        ),
        Turn(
            "The package seems delayed. I'm considering a return. "
            "Is this order eligible for a refund?",
            assertions=_turn_health_check,
        ),
        Turn(
            input=adaptive_query,
            assertions=_turn_health_check,
        ),
        Turn(
            "I've had enough of these automated responses. "
            "Please escalate this to a human support agent with high priority.",
            assertions=_turn_health_check,
        ),
    ],
    assertions=[
        AllTurnsComplete(allow_failures=2),  # Allow some failures for fuzz testing
        CompletesWithin(300.0),  # 5 minutes for complex scenario
        MaxTotalLLMCalls(30),
        as_assertion(get_error_handling_metric, name="error-handling-check"),
    ],
    tags=["complex"],
)

# Alternative baseline with frustrated customer flow
frustrated_baseline = Scenario(
    name="baseline-frustrated",
    description="Frustrated customer journey with dynamic escalation",
    agent=run_support_agent,
    turns=[
        Turn(input=frustrated_followup, assertions=_turn_health_check),
        Turn(input=frustrated_followup, assertions=_turn_health_check),
        Turn(input=frustrated_followup, assertions=_turn_health_check),
        Turn(input=frustrated_followup, assertions=_turn_health_check),
        Turn(input=frustrated_followup, assertions=_turn_health_check),
    ],
    assertions=[
        AllTurnsComplete(allow_failures=2),
        CompletesWithin(300.0),
        MaxTotalLLMCalls(35),
        as_assertion(get_error_handling_metric, name="error-handling-check"),
    ],
    tags=["frustrated"],
)


# Default balanced fuzzing
default_space = ChaosSpace(
    llm=LLMFuzzConfig(probability=0.3),
    stream=StreamFuzzConfig.disabled(),  # we don't use streaming in this agent
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


fuzz_default = fuzz_chaos(complex_baseline, n=5, seed=42, space=default_space)
fuzz_tool_heavy = fuzz_chaos(complex_baseline, n=5, seed=43, space=tool_heavy_space)
fuzz_llm_heavy = fuzz_chaos(complex_baseline, n=3, seed=44, space=llm_heavy_space)
fuzz_stress = fuzz_chaos(frustrated_baseline, n=3, seed=45, space=stress_space)
fuzz_malformed = fuzz_chaos(complex_baseline, n=2, seed=46, space=malformed_space)


def get_scenarios() -> list[Scenario]:
    """Return all fuzz scenarios for discovery."""
    base_scenarios = [complex_baseline, frustrated_baseline]
    fuzz_scenarios = (
        fuzz_default + fuzz_tool_heavy + fuzz_llm_heavy + fuzz_stress + fuzz_malformed
    )
    return base_scenarios + fuzz_scenarios
