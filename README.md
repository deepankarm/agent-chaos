<div align="center">

# agent-chaos

<img src="src/agent_chaos/ui/static/favicon.svg" width="150" height="150" alt="agent-chaos logo">

**Chaos engineering for AI agents.**

> *"Introduce a little anarchy. Upset the established order, and everything becomes chaos. I'm an agent of chaos. Oh, and you know the thing about chaos? It's fair!"*

</div>

---

Your agent works in demos. It passes evals. Then it hits production: the LLM sends a 500, the tool returns garbage, the stream cuts mid-response. The agent fails silently, confidently returns wrong answers, or loops forever.

**agent-chaos** breaks your agent on purpose, before production does. For teams building agents for production, not demos.

```bash
pip install agent-chaos
```

---

## Why does this exist?

LLM APIs are unreliable. They claim certain rate limits, then behave differently. They accept a stream request, then start sending tokens 10 seconds later. They reject mid-stream. They hang for 20 seconds before returning a 500. We've seen providers return `"Sorry about that"` as an error message.

Production agent backends run multiple LLMs with retry and fallback because things break randomly. What worked last week might not work today. You often don't realize it until production.

But the chaos isn't just at the transport layer. There's a semantic layer that's harder to catch.

Tools fail in obvious ways (timeouts, errors), but also in subtle ways: empty responses, partial data, wrong data types, malformed JSON, stale information, or data for the wrong entity entirely. A tool might return a 200 OK with an error message buried in the response body. An LLM-backed tool might hallucinate. With MCP, your agent calls tools you don't control, with schemas that can change without notice.

Traditional chaos engineering tools (Chaos Monkey, Gremlin) operate at infrastructure: network partitions, pod failures. They can't corrupt a tool result or cut an LLM stream mid-response.

**agent-chaos** injects these failures so you can test how your agent handles them before users find out. It integrates with evaluation frameworks like DeepEval, so you can inject chaos and judge the quality of your agent's response.

---

## Core concepts

### Scenarios: baseline + variants

A baseline scenario defines a conversation with your agent. A variant adds chaos:

```python
from agent_chaos import BaselineScenario, Turn
from agent_chaos.chaos import llm_rate_limit, tool_error

# Baseline: happy path
baseline = BaselineScenario(
    name="order-inquiry",
    agent=my_agent,
    turns=[
        Turn("What's the status of order #123?"),
        Turn("Can I get a refund?"),
    ],
)

# Variant: what happens when the LLM rate-limits?
baseline.variant(
    name="llm-rate-limit",
    chaos=[llm_rate_limit().after_calls(1)],
)

# Variant: what happens when the refund API fails?
baseline.variant(
    name="refund-api-down",
    chaos=[tool_error("Service unavailable").for_tool("check_refund")],
)
```

### Chaos and assertions

agent-chaos provides chaos injectors for LLM failures (`llm_rate_limit`, `llm_server_error`, `llm_timeout`), tool failures (`tool_error`, `tool_timeout`), data corruption (`tool_mutate`), and more. These are composable and support targeting specific tools, turns, or call counts.

Built-in assertions include `MaxTotalLLMCalls`, `AllTurnsComplete`, `TokenBurstDetection`, among others. For semantic evaluation, agent-chaos optionally integrates with [DeepEval](https://github.com/confident-ai/deepeval), letting you use any DeepEval metric (like `GEval`) as an assertion.

Both chaos and assertions can be applied per-scenario or per-turn using the `at()` helper:

```python
from agent_chaos import at
from agent_chaos.chaos import tool_error
from agent_chaos.scenario import CompletesWithin
from agent_chaos.integrations.deepeval import as_assertion
from deepeval.metrics import GEval

# Inject chaos only on turn 2
baseline.variant(
    name="check-refund-fails",
    turns=[
        at(
            2, 
            chaos=[tool_error("Service unavailable").for_tool("check_refund")],
            assertions=[
                CompletesWithin(60.0),
                as_assertion(GEval(name="task-completion", criteria="Did the agent complete the user's request?")),
            ],
        ),
    ],
)
```

---

## Fuzzing

It's difficult to define every failure mode upfront. `fuzz_chaos` generates random chaos combinations based on a `ChaosSpace` configuration, so you can explore how your agent behaves under varied conditions.

```python
from agent_chaos import fuzz_chaos, ChaosSpace, LLMFuzzConfig, ToolFuzzConfig

variants = fuzz_chaos(
    baseline, 
    n=10, 
    space=ChaosSpace(
        llm=LLMFuzzConfig(probability=0.3),
        tool=ToolFuzzConfig(probability=0.5, targets=["get_order", "process_refund"]),
    ),
)
```

Fuzzing is for exploration, not CI. See [`examples/ecommerce-support-agent/scenarios/fuzzing.py`](examples/ecommerce-support-agent/scenarios/fuzzing.py) for more.

---

## Examples

The [`examples/ecommerce-support-agent/`](examples/ecommerce-support-agent/) directory contains a complete example with an e-commerce support agent built with `pydantic-ai`, including:

- [`scenarios/quickstart.py`](examples/ecommerce-support-agent/scenarios/quickstart.py) - baseline scenarios with chaos variants
- [`scenarios/resilience.py`](examples/ecommerce-support-agent/scenarios/resilience.py) - comprehensive resilience testing
- [`scenarios/fuzzing.py`](examples/ecommerce-support-agent/scenarios/fuzzing.py) - automated, random chaos generation

```bash
cd examples/ecommerce-support-agent
uv sync
uv run agent-chaos run scenarios/quickstart.py

# on another terminal
uv run agent-chaos ui .agent_chaos_runs
```

**Scenario overview showing baselines, chaos variants, and assertion results:**

![All scenarios](.github/images/all%20scenarios.png)

**LLM rate limit injected on turn 1. Agent failed to respond, caught by turn-coherence assertion:**

![Rate limit injected](.github/images/rate-limit-injected.png)

**Tool error injected. Agent gracefully handles the failure and offers alternatives:**

![Tool error injected](.github/images/tool-error-injected.png)

---

## Status

Under active development.

**Supported:**
- Anthropic models (via `anthropic` SDK)
- Multi-turn conversations
- LLM chaos (rate limits, server errors, timeouts, stream cut/hang, slow chunks)
- Tool chaos (errors, timeouts, mutations)
- User input chaos (prompt injections)
- Optional DeepEval integration for LLM-as-judge assertions
- Scenario fuzzing

**Planned:**
- OpenAI, Gemini models
- Integration with other evaluation tools
- More chaos types
