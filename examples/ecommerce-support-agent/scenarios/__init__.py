"""E-commerce Support Agent Chaos Scenarios.

Chaos Engineering for AI Agents: 10 production-critical scenarios that answer
business questions, not just test technical capabilities.

Categories:
- LLM Call Chaos (3): Rate limits, 500 errors, call bounds
- LLM Stream Chaos (0): [TODO - needs streaming agent implementation]
- Tool Chaos (5): Errors, timeouts, semantic corruption, retry storms, cascades
- User Input Chaos (2): Prompt injection, user frustration

Key features:
- LLM-powered semantic mutations for tool data corruption
- LLM-generated dynamic user inputs for multi-turn scenarios
- Each scenario maps to a real production incident

Each scenario maps to a real production incident that would bite an enterprise.
"""

from .chaos_scenarios import (
    chaos_scenarios,
    llm_call_scenarios,
    stream_chaos_scenarios,
    tool_chaos_scenarios,
    user_input_scenarios,
)


def get_scenarios():
    """Return all 10 chaos engineering scenarios.

    These scenarios test:
    1. LLM rate limit recovery - Does retry/fallback work?
    2. LLM 500 error handling - Graceful message or raw error?
    3. LLM call bounds - Cost control, no spiraling
    4. Tool error no false promises - Don't claim success on failure
    5. Tool timeout bounded wait - Don't wait forever
    6. Tool semantic corruption - Catch LLM-generated bad data
    7. Tool retry storm prevention - Bound retries when user insists (LLM-generated)
    8. Tool cascading failure - Escalate on major outage
    9. Adversarial prompt injection - Stay on topic, no leaks
    10. User frustration handling - Maintain professionalism (LLM-generated)

    Stream chaos scenarios (TTFT, hang) require streaming agent - TODO.
    """
    return chaos_scenarios


__all__ = [
    "get_scenarios",
    "chaos_scenarios",
    "llm_call_scenarios",
    "stream_chaos_scenarios",
    "tool_chaos_scenarios",
    "user_input_scenarios",
]
