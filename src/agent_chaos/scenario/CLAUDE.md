# Scenario Module

Test scenario framework.

## Scenario Types

- **BaselineScenario**: Agent with no chaos - establish baseline behavior
- **ChaosScenario**: Agent with chaos injection - test resilience

## Execution Flow

```
run_scenario(scenario)
    ├── Create ChaosContext with injector, recorder, metrics
    ├── Patch LLM provider (anthropic, etc.)
    ├── For each Turn:
    │   ├── ctx.start_turn(n, input)
    │   ├── Call agent(ctx, turn_input)
    │   ├── ctx.end_turn() → TurnResult
    │   └── Run turn-level assertions
    ├── Unpatch provider
    ├── Run scenario-level assertions
    └── Return RunReport
```

## Turn Mechanics

A "turn" is one user input → agent response cycle. The framework tracks:
- Turn number (1-indexed)
- LLM calls made during the turn
- Tokens consumed
- Duration
- Success/failure

```python
# Turns are defined in the scenario
turns = [
    Turn("What's the weather?"),           # Turn 1
    Turn("Now book a flight"),             # Turn 2
    Turn(lambda ctx: f"Got {ctx.turn_results[-1].response}"),  # Dynamic
]
```

## Assertions

Scenario-level (run after all turns):
- `CompletesWithin(timeout_s)` - Total time
- `MaxLLMCalls(n)` / `MinLLMCalls(n)` - Call counts
- `MaxTokens(n)` - Token limits
- `AllTurnsComplete()` - All turns pass

Turn-level (check specific turns):
- `TurnCompletes(turn=1)` - Specific turn success
- `TurnCompletesWithin(timeout_s, turn=1)` - Turn timing
- `TurnResponseContains(substring, turn=1)` - Response content

## Usage

```python
from agent_chaos.scenario import ChaosScenario, Turn, run_scenario

scenario = ChaosScenario(
    name="rate-limit-recovery",
    agent=my_agent,
    turns=[Turn("Hello"), Turn("Do something")],
    chaos=[LLMChaos.rate_limit(on=on_turn(0))],
    assertions=[CompletesWithin(30.0), MaxLLMCalls(5)],
)

report = run_scenario(scenario)
assert report.passed
```
