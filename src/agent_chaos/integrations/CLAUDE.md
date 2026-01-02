# Integrations Module

DeepEval integration for LLM-as-judge evaluation.

## DeepEval (`deepeval.py`)

Wraps DeepEval metrics as agent-chaos assertions.

### Main API

```python
from agent_chaos.integrations.deepeval import (
    as_assertion,              # Convenience wrapper (auto-detects metric type)
    DeepEvalAssertion,         # Single-turn LLMTestCase metrics
    ConversationalDeepEvalAssertion,  # Multi-turn ConversationalTestCase metrics
    build_llm_test_case,       # Build LLMTestCase from ChaosContext
    build_conversational_test_case,   # Build ConversationalTestCase from ChaosContext
)
```

### Usage

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from agent_chaos.integrations.deepeval import as_assertion

scenario = ChaosScenario(
    name="test",
    agent=my_agent,
    assertions=[
        as_assertion(GEval(
            name="task-completion",
            criteria="Did the agent complete the task?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )),
    ],
)
```

### Helper Functions

- `_extract_tools_called(ctx)` - Tool calls as DeepEval ToolCall objects
- `_extract_retrieval_context(ctx)` - Tool results as retrieval context
- `_extract_chaos_context(ctx)` - Chaos injection info for evaluators
- `_format_conversation_for_eval(ctx)` - Multi-turn conversation formatting

### Requirements

DeepEval is an optional dependency: `pip install deepeval`
