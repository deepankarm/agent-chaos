"""Shared test fixtures for agent-chaos tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from agent_chaos.chaos.base import ChaosResult, TriggerConfig
from agent_chaos.chaos.llm import RateLimitChaos, ServerErrorChaos, TimeoutChaos
from agent_chaos.chaos.stream import SlowTTFTChaos, StreamCutChaos, StreamHangChaos
from agent_chaos.chaos.tool import ToolErrorChaos, ToolMutateChaos
from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore
from agent_chaos.core.recorder import Recorder

if TYPE_CHECKING:
    from agent_chaos.core.context import ChaosContext


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture
def metrics_store() -> MetricsStore:
    """Fresh MetricsStore instance."""
    return MetricsStore()


@pytest.fixture
def recorder(metrics_store: MetricsStore) -> Recorder:
    """Fresh Recorder instance with metrics store."""
    return Recorder(metrics=metrics_store)


@pytest.fixture
def chaos_injector() -> ChaosInjector:
    """Fresh ChaosInjector with no chaos configured."""
    return ChaosInjector(chaos=[])


@pytest.fixture
def chaos_context(chaos_injector: ChaosInjector, recorder: Recorder) -> ChaosContext:
    """Fresh ChaosContext with injector and recorder."""
    from agent_chaos.core.context import ChaosContext

    return ChaosContext(
        name="test-context",
        injector=chaos_injector,
        recorder=recorder,
        session_id="test-session-123",
    )


# =============================================================================
# Trigger Config Fixtures
# =============================================================================


@pytest.fixture
def trigger_on_call_2() -> TriggerConfig:
    """TriggerConfig that fires on call 2."""
    return TriggerConfig(on_call=2)


@pytest.fixture
def trigger_after_calls_3() -> TriggerConfig:
    """TriggerConfig that fires after 3 calls."""
    return TriggerConfig(after_calls=3)


@pytest.fixture
def trigger_on_turn_2() -> TriggerConfig:
    """TriggerConfig that fires on turn 2."""
    return TriggerConfig(on_turn=2)


@pytest.fixture
def trigger_always() -> TriggerConfig:
    """TriggerConfig that always fires."""
    return TriggerConfig(always=True)


@pytest.fixture
def trigger_probability_50() -> TriggerConfig:
    """TriggerConfig with 50% probability."""
    return TriggerConfig(probability=0.5)


@pytest.fixture
def trigger_for_anthropic() -> TriggerConfig:
    """TriggerConfig that only fires for anthropic provider."""
    return TriggerConfig(always=True, provider="anthropic")


# =============================================================================
# LLM Chaos Fixtures
# =============================================================================


@pytest.fixture
def rate_limit_chaos() -> RateLimitChaos:
    """RateLimitChaos that triggers on first call."""
    return RateLimitChaos(on_call=1)


@pytest.fixture
def timeout_chaos() -> TimeoutChaos:
    """TimeoutChaos that triggers on first call."""
    return TimeoutChaos(on_call=1, timeout_seconds=5.0)


@pytest.fixture
def server_error_chaos() -> ServerErrorChaos:
    """ServerErrorChaos that triggers on first call."""
    return ServerErrorChaos(on_call=1)


# =============================================================================
# Stream Chaos Fixtures
# =============================================================================


@pytest.fixture
def stream_cut_chaos() -> StreamCutChaos:
    """StreamCutChaos that cuts after 5 chunks."""
    return StreamCutChaos(after_chunks=5)


@pytest.fixture
def stream_hang_chaos() -> StreamHangChaos:
    """StreamHangChaos that hangs after 10 chunks."""
    return StreamHangChaos(after_chunks=10, hang_seconds=2.0)


@pytest.fixture
def slow_ttft_chaos() -> SlowTTFTChaos:
    """SlowTTFTChaos with 1 second delay."""
    return SlowTTFTChaos(delay=1.0)


# =============================================================================
# Tool Chaos Fixtures
# =============================================================================


@pytest.fixture
def tool_error_chaos() -> ToolErrorChaos:
    """ToolErrorChaos that triggers for any tool."""
    return ToolErrorChaos(error_message="Tool failed", always=True)


@pytest.fixture
def tool_mutate_chaos() -> ToolMutateChaos:
    """ToolMutateChaos that mutates tool results."""

    def mutator(tool_name: str, result: str) -> str:
        return f"MUTATED: {result}"

    return ToolMutateChaos(mutator=mutator, always=True)


# =============================================================================
# Injector with Chaos Fixtures
# =============================================================================


@pytest.fixture
def injector_with_rate_limit(rate_limit_chaos: RateLimitChaos) -> ChaosInjector:
    """ChaosInjector with rate limit chaos configured."""
    return ChaosInjector(chaos=[rate_limit_chaos])


@pytest.fixture
def injector_with_stream_chaos(
    stream_cut_chaos: StreamCutChaos,
    slow_ttft_chaos: SlowTTFTChaos,
) -> ChaosInjector:
    """ChaosInjector with stream chaos configured."""
    return ChaosInjector(chaos=[stream_cut_chaos, slow_ttft_chaos])


@pytest.fixture
def injector_with_multiple_chaos(
    rate_limit_chaos: RateLimitChaos,
    stream_cut_chaos: StreamCutChaos,
    tool_error_chaos: ToolErrorChaos,
) -> ChaosInjector:
    """ChaosInjector with multiple chaos types configured."""
    return ChaosInjector(chaos=[rate_limit_chaos, stream_cut_chaos, tool_error_chaos])


# =============================================================================
# Mock Provider Fixtures
# =============================================================================


@pytest.fixture
def mock_anthropic_message() -> dict[str, Any]:
    """Mock Anthropic Message response structure."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, I'm Claude!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


@pytest.fixture
def mock_anthropic_tool_use() -> dict[str, Any]:
    """Mock Anthropic tool_use response."""
    return {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "Tokyo"},
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 15, "output_tokens": 25},
    }


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mocked Anthropic client for patcher tests."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = MagicMock()
    client.messages.stream = MagicMock()
    return client


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mocked OpenAI client for patcher tests."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client


# =============================================================================
# Scenario Fixtures
# =============================================================================


@pytest.fixture
def sample_agent() -> Any:
    """Simple mock agent callable for testing scenarios."""

    def agent(ctx: ChaosContext, turn_input: str) -> str:
        return f"Response to: {turn_input}"

    return agent


@pytest.fixture
def sample_baseline_scenario(sample_agent: Any) -> Any:
    """Sample BaselineScenario for testing."""
    from agent_chaos.scenario.model import BaselineScenario, Turn

    return BaselineScenario(
        name="test-baseline",
        description="A test baseline scenario",
        agent=sample_agent,
        turns=[
            Turn("Hello"),
            Turn("How are you?"),
            Turn("Goodbye"),
        ],
    )


@pytest.fixture
def sample_chaos_scenario(sample_baseline_scenario: Any, rate_limit_chaos: RateLimitChaos) -> Any:
    """Sample ChaosScenario created from baseline."""
    return sample_baseline_scenario.variant(
        name="test-chaos",
        chaos=[rate_limit_chaos],
    )


# =============================================================================
# Assertion Result Fixtures
# =============================================================================


@pytest.fixture
def passing_assertion_result() -> Any:
    """Sample passing AssertionResult."""
    from agent_chaos.scenario.assertions import AssertionResult

    return AssertionResult(
        name="test_assertion",
        passed=True,
        message="Assertion passed",
    )


@pytest.fixture
def failing_assertion_result() -> Any:
    """Sample failing AssertionResult."""
    from agent_chaos.scenario.assertions import AssertionResult

    return AssertionResult(
        name="test_assertion",
        passed=False,
        message="Assertion failed: expected X but got Y",
    )
