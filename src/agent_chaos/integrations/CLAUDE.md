# Integrations Module

LLM-as-judge evaluation integrations for agent-chaos.

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

## Pydantic Evals (`pydantic_evals.py`)

Wraps pydantic-evals evaluators (especially LLMJudge) as agent-chaos assertions.

### Main API

```python
from agent_chaos.integrations.pydantic_evals import (
    as_assertion,              # Wrap any pydantic-evals Evaluator
    PydanticEvalsAssertion,    # Underlying wrapper class
    build_evaluator_context,   # Build EvaluatorContext from ChaosContext
)
```

### Usage

```python
from pydantic_evals.evaluators import LLMJudge
from agent_chaos.integrations.pydantic_evals import as_assertion

scenario = ChaosScenario(
    name="test",
    agent=my_agent,
    assertions=[
        as_assertion(
            LLMJudge(
                rubric="Agent handles errors gracefully and informs the user",
                model="anthropic:claude-sonnet-4-5",
                include_input=True,
            ),
            threshold=0.7,
        ),
    ],
)
```

### Parameters

- `evaluator` - Any pydantic-evals Evaluator (LLMJudge, custom Evaluator subclass)
- `threshold` - Score threshold for pass/fail (default: 0.5 for score-based evaluators)
- `expected_output` - Expected output for comparison evaluators
- `turn` - Specific turn to evaluate (1-indexed)
- `name` - Custom assertion name
- `include_chaos_info` - Include chaos injection info in evaluator context (default: True)

### Custom Evaluators

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.context import EvaluatorContext

@dataclass
class ResponseContainsKeyword(Evaluator[str, str, None]):
    keyword: str

    def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
        return self.keyword.lower() in ctx.output.lower()

assertion = as_assertion(ResponseContainsKeyword(keyword="weather"))
```

### Requirements

Pydantic Evals is an optional dependency: `pip install pydantic-evals`
