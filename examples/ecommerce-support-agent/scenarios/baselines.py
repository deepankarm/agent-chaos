"""Baseline experiments for the e-commerce support agent.

This module defines the two protagonist experiments that all variants build upon:

1. `customer_journey` - A structured 5-turn conversation:
   - Turn 0: Order status inquiry
   - Turn 1: Shipping details request
   - Turn 2: Refund eligibility check
   - Turn 3: Adaptive follow-up (LLM-generated based on previous response)
   - Turn 4: Escalation to human support

2. `frustrated_customer` - A dynamic 5-turn escalation:
   - All turns are LLM-generated with increasing frustration
   - Tests agent's ability to handle emotional escalation

Both baselines include:
- Per-turn coherence checks (response quality)
- Scenario-level error handling evaluation
- DeepEval LLM-as-judge metrics

These baselines are imported by quickstart.py, resilience.py, and fuzzing.py
to create variants with different chaos injections.
"""

from agent_chaos import Turn
from agent_chaos.scenario import (
    AllTurnsComplete,
    CompletesWithin,
    MaxTotalLLMCalls,
    Scenario,
)

from agent import run_support_agent

from .commons import (
    adaptive_query,
    error_handling,
    frustrated_followup,
    turn_coherence,
)

# =============================================================================
# Baseline Experiments
# =============================================================================

# Per-turn health check
_turn_health_check = [turn_coherence]

# Baseline 1: Structured 5-turn customer journey
customer_journey = Scenario(
    name="customer-journey",
    description="5-turn journey: order status → shipping → refund check → follow-up → escalation",
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
        AllTurnsComplete(allow_failures=1),
        CompletesWithin(300.0),
        MaxTotalLLMCalls(30),
        error_handling,
    ],
    tags=["baseline"],
)

# Baseline 2: Frustrated customer with dynamic escalation
frustrated_customer = Scenario(
    name="frustrated-customer",
    description="5-turn dynamic escalation with increasingly frustrated customer (LLM-generated)",
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
        error_handling,
    ],
    tags=["baseline", "dynamic"],
)
