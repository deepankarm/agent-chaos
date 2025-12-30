# agent-chaos

**Chaos engineering for AI agents.**

> The Joker: *"Introduce a little anarchy. Upset the established order, and everything becomes chaos. I'm an agent of chaos. Oh, and you know the thing about chaos? It's fair!"*

---

Your agent works in demos. It passes evals. Then it hits production: the LLM rate-limits, the tool API returns garbage, the stream cuts mid-response. The agent fails silently, confidently returns wrong answers, or loops forever.

**agent-chaos** breaks your agent on purpose—before production does.

---

## Why This Exists

AI agents have failure boundaries that didn't exist before:

| Boundary | What Can Break |
|----------|----------------|
| **LLM provider** | Rate limits, timeouts, server errors, stream interruptions |
| **Tool execution** | API failures, malformed responses, lies |
| **Context/memory** | Corrupted retrieval, poisoned history, token overflow |

Traditional chaos engineering tools (Chaos Monkey, Gremlin, Litmus) operate at the infrastructure layer—network partitions, pod failures, CPU stress. They don't understand agent-specific failure modes. They can't corrupt a tool result or cut an LLM stream after 10 chunks.

Evaluation tools (Galileo, DeepEval, LangSmith) tell you *if* your agent worked correctly. They judge past runs. They can't answer: *"What happens when the weather API lies?"*

**agent-chaos injects failures. Eval tools judge outcomes. Use both.**

```
                     ┌─────────────────┐
                     │  agent-chaos    │
                     │  (inject chaos) │
                     └────────┬────────┘
                              │
                              ▼
┌──────────────┐       ┌─────────────┐       ┌──────────────┐
│   CI / Test  │──────▶│  Your Agent │──────▶│  Eval Tools  │
│   Pipeline   │       │             │       │  (judge it)  │
└──────────────┘       └─────────────┘       └──────────────┘
```

---

## What It Does

Inject chaos at every agent boundary:

```python
from agent_chaos import (
    chaos_context,
    llm_rate_limit,
    llm_stream_cut,
    tool_error,
    tool_mutate,
)

def corrupt_weather(tool_name: str, result: str) -> str:
    # Return plausible lies
    return result.replace("22°C", "-50°C")

with chaos_context(
    name="resilience-test",
    chaos=[
        llm_rate_limit().after_calls(2),
        llm_stream_cut(after_chunks=10).with_probability(0.3),
        tool_error("Service unavailable").for_tool("get_weather").on_call(1),
        tool_mutate(corrupt_weather),
    ],
) as ctx:
    response = my_agent.run("What's the weather in Tokyo?")

    # Did the agent handle the chaos?
    assert ctx.metrics.chaos_injected > 0
```

Then gate your CI:

```bash
agent-chaos run scenarios/ --artifacts-dir ./runs
# Exit code 0 = all scenarios passed
# Exit code 1 = failures detected
```

---

## The Two Questions

| Question | Tool |
|----------|------|
| *"Did my agent give the right answer?"* | Eval tools (Galileo, DeepEval, LangSmith) |
| *"Did my agent survive when dependencies failed?"* | **agent-chaos** |

Evals test correctness. Chaos tests resilience. Production needs both.

---

## What About Edge-Case Inputs?

"What if the user asks something unexpected?"

This is a fair question, but it's not chaos engineering—it's evaluation. A weird user query isn't a *fault*. It's just input. The user isn't failing; they're being a user.

For testing agent behavior on edge-case inputs: use eval tools with golden datasets. That's what they're built for.

**agent-chaos** is for failures at *external dependencies*—the LLM provider, tool APIs, memory systems. Things that break independently of what the user asked.

The exception: **multi-agent systems**. When Agent A's output becomes Agent B's input, corrupted handoffs *are* faults at a boundary. That's chaos territory.

---

## Who This Is For

Teams shipping agents to production. Not demos. Not prototypes.

If you've been burned by:
- Silent failures from flaky tool APIs
- Agents that loop forever on rate limits
- Confident wrong answers from corrupted context
- "Works on my machine" syndrome in agent behavior

This is for you.

---

## Status

Under active development. Anthropic provider supported. OpenAI and Gemini planned.

