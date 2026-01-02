"""Tests for core/metrics/ - MetricsStore class."""

from __future__ import annotations

import time

import pytest

from agent_chaos.core.metrics import CallRecord, MetricsStore


@pytest.fixture
def metrics() -> MetricsStore:
    """Fresh MetricsStore for testing."""
    return MetricsStore()


class TestMetricsStoreInit:
    """Tests for MetricsStore initialization."""

    def test_default_values(self, metrics: MetricsStore) -> None:
        assert metrics.calls.count == 0
        assert metrics.calls.retries == 0
        assert metrics.calls.latencies == []
        assert metrics.faults == []
        assert metrics.history == []
        assert metrics.conv.entries == []

    def test_total_calls_property(self, metrics: MetricsStore) -> None:
        assert metrics.total_calls == 0
        metrics.calls.count = 5
        assert metrics.total_calls == 5


class TestMetricsStoreCallTracking:
    """Tests for call start/end tracking."""

    def test_start_call(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        assert call_id.startswith("anthropic_")
        assert metrics.calls.count == 1
        assert "anthropic" in metrics.calls.by_provider
        assert metrics.calls.by_provider["anthropic"] == 1

    def test_multiple_start_calls(self, metrics: MetricsStore) -> None:
        metrics.start_call("anthropic")
        metrics.start_call("anthropic")
        metrics.start_call("openai")
        assert metrics.calls.count == 3
        assert metrics.calls.by_provider["anthropic"] == 2
        assert metrics.calls.by_provider["openai"] == 1

    def test_end_call_success(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        time.sleep(0.01)  # Small delay for latency
        metrics.end_call(call_id, success=True)

        assert len(metrics.history) == 1
        call = metrics.history[0]
        assert call.call_id == call_id
        assert call.provider == "anthropic"
        assert call.success is True
        assert call.latency > 0
        assert call.error is None

    def test_end_call_failure(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        error = Exception("Rate limit exceeded")
        metrics.end_call(call_id, success=False, error=error)

        assert len(metrics.history) == 1
        call = metrics.history[0]
        assert call.success is False
        assert "Rate limit" in call.error

    def test_end_call_tracks_retries(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        error = Exception("429 rate limit")
        metrics.end_call(call_id, success=False, error=error)
        assert metrics.calls.retries == 1

    def test_end_call_adds_latency_on_success(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        time.sleep(0.01)
        metrics.end_call(call_id, success=True)
        assert len(metrics.calls.latencies) == 1
        assert metrics.calls.latencies[0] > 0

    def test_end_call_nonexistent(self, metrics: MetricsStore) -> None:
        # Should not raise
        metrics.end_call("nonexistent-call", success=True)
        assert len(metrics.history) == 0


class TestMetricsStoreLatency:
    """Tests for latency tracking."""

    def test_avg_latency_empty(self, metrics: MetricsStore) -> None:
        assert metrics.avg_latency == 0.0

    def test_avg_latency_single(self, metrics: MetricsStore) -> None:
        metrics.calls.latencies = [1.0]
        assert metrics.avg_latency == 1.0

    def test_avg_latency_multiple(self, metrics: MetricsStore) -> None:
        metrics.calls.latencies = [1.0, 2.0, 3.0]
        assert metrics.avg_latency == 2.0


class TestMetricsStoreSuccessRate:
    """Tests for success rate calculation."""

    def test_success_rate_no_calls(self, metrics: MetricsStore) -> None:
        assert metrics.success_rate == 1.0

    def test_success_rate_all_success(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(call_id="1", provider="test", success=True, latency=0.1),
            CallRecord(call_id="2", provider="test", success=True, latency=0.1),
            CallRecord(call_id="3", provider="test", success=True, latency=0.1),
        ]
        assert metrics.success_rate == 1.0

    def test_success_rate_all_failure(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(call_id="1", provider="test", success=False, latency=0.1),
            CallRecord(call_id="2", provider="test", success=False, latency=0.1),
        ]
        assert metrics.success_rate == 0.0

    def test_success_rate_mixed(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(call_id="1", provider="test", success=True, latency=0.1),
            CallRecord(call_id="2", provider="test", success=False, latency=0.1),
            CallRecord(call_id="3", provider="test", success=True, latency=0.1),
            CallRecord(call_id="4", provider="test", success=True, latency=0.1),
        ]
        assert metrics.success_rate == 0.75


class TestMetricsStoreTokenTracking:
    """Tests for token tracking."""

    def test_total_input_tokens(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 200},
            ),
        ]
        assert metrics.total_input_tokens == 300

    def test_total_output_tokens(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"output_tokens": 50},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"output_tokens": 100},
            ),
        ]
        assert metrics.total_output_tokens == 150

    def test_total_tokens(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 200, "output_tokens": 100},
            ),
        ]
        assert metrics.total_tokens == 450

    def test_avg_tokens_per_call_empty(self, metrics: MetricsStore) -> None:
        assert metrics.avg_tokens_per_call == 0.0

    def test_avg_tokens_per_call(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 200, "output_tokens": 100},
            ),
        ]
        assert metrics.avg_tokens_per_call == 225.0  # (150 + 300) / 2

    def test_max_tokens_single_call(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            CallRecord(
                call_id="2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 500, "output_tokens": 300},
            ),
            CallRecord(
                call_id="3",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 200, "output_tokens": 100},
            ),
        ]
        assert metrics.max_tokens_single_call == 800  # 500 + 300

    def test_max_tokens_single_call_empty(self, metrics: MetricsStore) -> None:
        assert metrics.max_tokens_single_call == 0

    def test_tokens_with_missing_usage(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100},
            ),
            CallRecord(
                call_id="2", provider="test", success=True, latency=0.1, usage={}
            ),
            CallRecord(call_id="3", provider="test", success=True, latency=0.1),
        ]
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 0

    def test_record_token_usage(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        metrics.record_token_usage(
            call_id,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="claude-3-opus",
        )
        # Usage should be stored in the active call
        call_info = metrics.get_active_call(call_id)
        assert call_info is not None
        assert call_info.usage["input_tokens"] == 100
        assert call_info.usage["output_tokens"] == 50


class TestMetricsStoreTurnTracking:
    """Tests for turn-related tracking."""

    def test_set_current_turn(self, metrics: MetricsStore) -> None:
        metrics.set_current_turn(3)
        assert metrics.conv.current_turn == 3


class TestMetricsStoreConversation:
    """Tests for conversation tracking."""

    def test_add_conversation_entry(self, metrics: MetricsStore) -> None:
        metrics.add_conversation_entry("user", content="Hello")
        assert len(metrics.conv.entries) == 1
        entry = metrics.conv.entries[0]
        assert entry["type"] == "user"
        assert entry["content"] == "Hello"
        assert "timestamp_ms" in entry

    def test_add_conversation_entry_with_turn_number(
        self, metrics: MetricsStore
    ) -> None:
        metrics.set_current_turn(2)
        metrics.add_conversation_entry("tool_call", tool_name="weather")
        entry = metrics.conv.entries[0]
        assert entry["turn_number"] == 2

    def test_user_message_deduplication(self, metrics: MetricsStore) -> None:
        metrics.add_conversation_entry("user", content="Hello")
        metrics.add_conversation_entry("user", content="Hello again")
        # Second user message should be skipped
        assert len(metrics.conv.entries) == 1

    def test_user_message_flag_reset(self, metrics: MetricsStore) -> None:
        metrics.add_conversation_entry("user", content="Hello")
        metrics.conv.user_message_recorded = False
        metrics.add_conversation_entry("user", content="Next turn")
        assert len(metrics.conv.entries) == 2


class TestMetricsStoreSystemPrompt:
    """Tests for system prompt recording."""

    def test_record_system_prompt_string(self, metrics: MetricsStore) -> None:
        metrics.record_system_prompt("You are a helpful assistant.")
        assert metrics.conv.system_prompt == "You are a helpful assistant."
        assert metrics.conv.system_prompt_recorded is True

    def test_record_system_prompt_only_once(self, metrics: MetricsStore) -> None:
        metrics.record_system_prompt("First prompt")
        metrics.record_system_prompt("Second prompt")
        assert metrics.conv.system_prompt == "First prompt"

    def test_record_system_prompt_none(self, metrics: MetricsStore) -> None:
        metrics.record_system_prompt(None)
        assert metrics.conv.system_prompt is None

    def test_record_system_prompt_list_format(self, metrics: MetricsStore) -> None:
        # Anthropic uses list format sometimes
        metrics.record_system_prompt(
            [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ]
        )
        assert "You are helpful." in metrics.conv.system_prompt
        assert "Be concise." in metrics.conv.system_prompt

    def test_record_system_prompt_adds_to_conversation(
        self, metrics: MetricsStore
    ) -> None:
        metrics.record_system_prompt("System prompt text")
        # System prompt is inserted at beginning of conversation
        assert len(metrics.conv.entries) == 1
        assert metrics.conv.entries[0]["type"] == "system"
        assert metrics.conv.entries[0]["content"] == "System prompt text"


class TestMetricsStoreFaults:
    """Tests for fault recording."""

    def test_record_fault(self, metrics: MetricsStore) -> None:
        metrics.record_fault("call-123", "RateLimitError", "anthropic")
        assert len(metrics.faults) == 1
        assert metrics.faults[0].call_id == "call-123"
        assert metrics.faults[0].fault_type == "RateLimitError"

    def test_record_fault_with_chaos_details(self, metrics: MetricsStore) -> None:
        metrics.record_fault(
            "call-123",
            "tool_error",
            "anthropic",
            chaos_point="TOOL",
            target_tool="weather",
            original='{"temp": 72}',
            mutated='{"error": "timeout"}',
        )
        # Should add to conversation
        chaos_entry = None
        for entry in metrics.conv.entries:
            if entry["type"] == "chaos":
                chaos_entry = entry
                break
        assert chaos_entry is not None
        assert chaos_entry.get("target_tool") == "weather"


class TestMetricsStoreToolTracking:
    """Tests for tool use tracking."""

    def test_record_tool_use(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        metrics.record_tool_use(
            call_id,
            tool_name="weather",
            tool_use_id="tool-123",
            input_bytes=50,
        )
        # Tool use should be recorded in active call
        call_info = metrics.get_active_call(call_id)
        assert call_info is not None
        assert len(call_info.tool_uses) == 1
        assert call_info.tool_uses[0]["tool_name"] == "weather"

    def test_record_tool_start(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        metrics.record_tool_start(
            tool_name="calculator",
            tool_use_id="tool-456",
            call_id=call_id,
        )
        assert "tool-456" in metrics.tools.started_at

    def test_record_tool_end(self, metrics: MetricsStore) -> None:
        metrics.record_tool_end(
            tool_name="calculator",
            success=True,
            tool_use_id="tool-456",
            duration_ms=150.0,
            result="42",
        )
        # Should add to conversation
        tool_result = None
        for entry in metrics.conv.entries:
            if entry["type"] == "tool_result":
                tool_result = entry
                break
        assert tool_result is not None
        assert tool_result["tool_name"] == "calculator"
        assert tool_result["success"] is True


class TestMetricsStoreStreamTracking:
    """Tests for stream-related tracking."""

    def test_record_ttft(self, metrics: MetricsStore) -> None:
        metrics.record_ttft(0.5, "call-123")
        assert len(metrics.stream.ttft_times) == 1
        assert metrics.stream.ttft_times[0] == 0.5

    def test_avg_ttft_empty(self, metrics: MetricsStore) -> None:
        assert metrics.avg_ttft == 0.0

    def test_avg_ttft(self, metrics: MetricsStore) -> None:
        metrics.stream.ttft_times = [0.1, 0.2, 0.3]
        assert abs(metrics.avg_ttft - 0.2) < 0.0001

    def test_record_hang(self, metrics: MetricsStore) -> None:
        metrics.record_hang(10, "call-123")
        assert len(metrics.stream.hang_events) == 1
        assert metrics.stream.hang_events[0] == 10

    def test_record_stream_cut(self, metrics: MetricsStore) -> None:
        metrics.record_stream_cut(5, "call-123")
        assert len(metrics.stream.stream_cuts) == 1
        assert metrics.stream.stream_cuts[0] == 5

    def test_record_chunk(self, metrics: MetricsStore) -> None:
        metrics.record_chunk(3)
        assert len(metrics.stream.chunk_counts) == 1
        assert metrics.stream.chunk_counts[0] == 3

    def test_record_stream_stats(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        metrics.record_stream_stats(call_id, chunk_count=25)
        call_info = metrics.get_active_call(call_id)
        assert call_info is not None
        assert call_info.stream_chunks == 25


class TestMetricsStoreTokenHistory:
    """Tests for token history."""

    def test_get_token_history(self, metrics: MetricsStore) -> None:
        metrics.history = [
            CallRecord(
                call_id="c1",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            CallRecord(
                call_id="c2",
                provider="test",
                success=True,
                latency=0.1,
                usage={"input_tokens": 200, "output_tokens": 100},
            ),
        ]
        history = metrics.get_token_history()
        assert len(history) == 2
        assert history[0]["total_tokens"] == 150
        assert history[0]["cumulative_tokens"] == 150
        assert history[1]["total_tokens"] == 300
        assert history[1]["cumulative_tokens"] == 450

    def test_get_token_history_empty(self, metrics: MetricsStore) -> None:
        history = metrics.get_token_history()
        assert history == []


class TestMetricsStoreWithFixtures:
    """Tests using conftest fixtures."""

    def test_initial_state(self, metrics_store: MetricsStore):
        """Test initial state values using fixture."""
        assert metrics_store.calls.count == 0
        assert metrics_store.calls.retries == 0
        assert metrics_store.calls.latencies == []
        assert metrics_store.faults == []
        assert metrics_store.history == []

    def test_initial_properties(self, metrics_store: MetricsStore):
        """Test initial computed properties using fixture."""
        assert metrics_store.total_calls == 0
        assert metrics_store.avg_latency == 0.0
        assert metrics_store.success_rate == 1.0
        assert metrics_store.total_input_tokens == 0
        assert metrics_store.total_output_tokens == 0


class TestMetricsStoreAccessorMethods:
    """Tests for new accessor methods."""

    def test_get_active_call(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        call_info = metrics.get_active_call(call_id)
        assert call_info is not None
        assert call_info.provider == "anthropic"

    def test_get_active_call_nonexistent(self, metrics: MetricsStore) -> None:
        call_info = metrics.get_active_call("nonexistent")
        assert call_info is None

    def test_get_cumulative_tokens(self, metrics: MetricsStore) -> None:
        call_id = metrics.start_call("anthropic")
        metrics.record_token_usage(call_id, input_tokens=100, output_tokens=50)
        input_tokens, output_tokens = metrics.get_cumulative_tokens()
        assert input_tokens == 100
        assert output_tokens == 50

    def test_register_tool_use(self, metrics: MetricsStore) -> None:
        metrics.register_tool_use("tool-1", "weather", "call-1")
        assert metrics.get_tool_name("tool-1") == "weather"
        assert metrics.tools.use_to_call_id["tool-1"] == "call-1"

    def test_is_tool_ended(self, metrics: MetricsStore) -> None:
        assert metrics.is_tool_ended("tool-1") is False
        metrics.mark_tool_ended("tool-1")
        assert metrics.is_tool_ended("tool-1") is True

    def test_reset_user_message_flag(self, metrics: MetricsStore) -> None:
        metrics.conv.user_message_recorded = True
        metrics.reset_user_message_flag()
        assert metrics.conv.user_message_recorded is False

    def test_is_tool_in_conversation(self, metrics: MetricsStore) -> None:
        assert metrics.is_tool_in_conversation("tool-1") is False
        metrics.mark_tool_in_conversation("tool-1")
        assert metrics.is_tool_in_conversation("tool-1") is True

    def test_get_tool_start_time(self, metrics: MetricsStore) -> None:
        metrics.record_tool_start(
            tool_name="test", tool_use_id="tool-1", call_id="call-1"
        )
        start_time = metrics.get_tool_start_time("tool-1")
        assert start_time is not None
        assert start_time > 0
