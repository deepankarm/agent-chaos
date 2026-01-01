"""Tests for LLM chaos types."""

from __future__ import annotations

import pytest

from agent_chaos.chaos.base import ChaosPoint
from agent_chaos.chaos.llm import (
    AuthErrorChaos,
    ContextLengthChaos,
    RateLimitChaos,
    ServerErrorChaos,
    TimeoutChaos,
    llm_auth_error,
    llm_context_length,
    llm_rate_limit,
    llm_server_error,
    llm_timeout,
)
from agent_chaos.types import ChaosAction


class TestRateLimitChaos:
    """Tests for RateLimitChaos."""

    def test_defaults(self) -> None:
        chaos = RateLimitChaos()
        assert chaos.retry_after == 30.0
        assert chaos.message == "Rate limit exceeded"
        assert chaos.on_call is None
        assert chaos.always is False

    def test_custom_retry_after(self) -> None:
        chaos = RateLimitChaos(retry_after=60.0)
        assert chaos.retry_after == 60.0

    def test_point(self) -> None:
        chaos = RateLimitChaos()
        assert chaos.point == ChaosPoint.LLM_CALL

    def test_should_trigger_on_call(self) -> None:
        chaos = RateLimitChaos(on_call=2)
        assert not chaos.should_trigger(1)
        assert chaos.should_trigger(2)
        assert not chaos.should_trigger(3)

    def test_should_trigger_always(self) -> None:
        chaos = RateLimitChaos(always=True)
        assert chaos.should_trigger(1)
        assert chaos.should_trigger(100)

    def test_should_trigger_after_calls(self) -> None:
        chaos = RateLimitChaos(after_calls=3)
        assert not chaos.should_trigger(1)
        assert not chaos.should_trigger(3)
        assert chaos.should_trigger(4)
        assert chaos.should_trigger(10)

    def test_apply_returns_raise_action(self) -> None:
        chaos = RateLimitChaos()
        result = chaos.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_to_exception_anthropic(self) -> None:
        import anthropic

        chaos = RateLimitChaos(message="Custom rate limit")
        exc = chaos.to_exception("anthropic")
        assert isinstance(exc, anthropic.RateLimitError)

    def test_to_exception_unsupported_provider(self) -> None:
        chaos = RateLimitChaos()
        with pytest.raises(NotImplementedError, match="Provider openai not implemented"):
            chaos.to_exception("openai")

    def test_str_representation(self) -> None:
        chaos = RateLimitChaos(retry_after=45.0, on_call=2)
        assert str(chaos) == "llm_rate_limit(45.0s) on call 2"

    def test_str_with_after_calls(self) -> None:
        chaos = RateLimitChaos(after_calls=5)
        assert str(chaos) == "llm_rate_limit(30.0s) after 5 calls"

    def test_str_with_probability(self) -> None:
        chaos = RateLimitChaos(probability=0.5)
        assert str(chaos) == "llm_rate_limit(30.0s) @50%"

    def test_provider_filter(self) -> None:
        chaos = RateLimitChaos(provider="anthropic", always=True)
        assert chaos.should_trigger(1, provider="anthropic")
        assert not chaos.should_trigger(1, provider="openai")


class TestTimeoutChaos:
    """Tests for TimeoutChaos."""

    def test_defaults(self) -> None:
        chaos = TimeoutChaos()
        assert chaos.timeout_seconds == 30.0
        assert chaos.message == "Request timed out"

    def test_custom_timeout(self) -> None:
        chaos = TimeoutChaos(timeout_seconds=120.0)
        assert chaos.timeout_seconds == 120.0

    def test_point(self) -> None:
        chaos = TimeoutChaos()
        assert chaos.point == ChaosPoint.LLM_CALL

    def test_apply_returns_raise_action(self) -> None:
        chaos = TimeoutChaos()
        result = chaos.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_to_exception_anthropic(self) -> None:
        import anthropic

        chaos = TimeoutChaos()
        exc = chaos.to_exception("anthropic")
        assert isinstance(exc, anthropic.APITimeoutError)

    def test_to_exception_unsupported_provider(self) -> None:
        chaos = TimeoutChaos()
        with pytest.raises(NotImplementedError):
            chaos.to_exception("openai")

    def test_str_representation(self) -> None:
        chaos = TimeoutChaos(timeout_seconds=60.0, on_call=3)
        assert str(chaos) == "llm_timeout(60.0s) on call 3"


class TestServerErrorChaos:
    """Tests for ServerErrorChaos."""

    def test_defaults(self) -> None:
        chaos = ServerErrorChaos()
        assert chaos.status_code == 500
        assert chaos.message == "Internal server error"

    def test_custom_status_code(self) -> None:
        chaos = ServerErrorChaos(status_code=502)
        assert chaos.status_code == 502

    def test_point(self) -> None:
        chaos = ServerErrorChaos()
        assert chaos.point == ChaosPoint.LLM_CALL

    def test_apply_returns_raise_action(self) -> None:
        chaos = ServerErrorChaos()
        result = chaos.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_to_exception_anthropic(self) -> None:
        import anthropic

        chaos = ServerErrorChaos(message="Custom error")
        exc = chaos.to_exception("anthropic")
        assert isinstance(exc, anthropic.InternalServerError)

    def test_to_exception_unsupported_provider(self) -> None:
        chaos = ServerErrorChaos()
        with pytest.raises(NotImplementedError):
            chaos.to_exception("openai")

    def test_str_representation(self) -> None:
        chaos = ServerErrorChaos(status_code=503, on_call=1)
        assert str(chaos) == "llm_server_error(503) on call 1"


class TestAuthErrorChaos:
    """Tests for AuthErrorChaos."""

    def test_defaults(self) -> None:
        chaos = AuthErrorChaos()
        assert chaos.message == "Invalid API key"

    def test_custom_message(self) -> None:
        chaos = AuthErrorChaos(message="API key expired")
        assert chaos.message == "API key expired"

    def test_point(self) -> None:
        chaos = AuthErrorChaos()
        assert chaos.point == ChaosPoint.LLM_CALL

    def test_apply_returns_raise_action(self) -> None:
        chaos = AuthErrorChaos()
        result = chaos.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_to_exception_anthropic(self) -> None:
        import anthropic

        chaos = AuthErrorChaos()
        exc = chaos.to_exception("anthropic")
        assert isinstance(exc, anthropic.AuthenticationError)

    def test_to_exception_unsupported_provider(self) -> None:
        chaos = AuthErrorChaos()
        with pytest.raises(NotImplementedError):
            chaos.to_exception("openai")

    def test_str_representation(self) -> None:
        chaos = AuthErrorChaos(on_call=5)
        assert str(chaos) == "llm_auth_error on call 5"


class TestContextLengthChaos:
    """Tests for ContextLengthChaos."""

    def test_defaults(self) -> None:
        chaos = ContextLengthChaos()
        assert chaos.max_tokens == 200000
        assert chaos.message == "Context length exceeded"

    def test_custom_max_tokens(self) -> None:
        chaos = ContextLengthChaos(max_tokens=100000)
        assert chaos.max_tokens == 100000

    def test_point(self) -> None:
        chaos = ContextLengthChaos()
        assert chaos.point == ChaosPoint.LLM_CALL

    def test_apply_returns_raise_action(self) -> None:
        chaos = ContextLengthChaos()
        result = chaos.apply(provider="anthropic")
        assert result.action == ChaosAction.RAISE
        assert result.exception is not None

    def test_to_exception_anthropic(self) -> None:
        chaos = ContextLengthChaos()
        exc = chaos.to_exception("anthropic")
        # BadRequestError or InvalidRequestError depending on SDK version
        assert exc is not None

    def test_to_exception_unsupported_provider(self) -> None:
        chaos = ContextLengthChaos()
        with pytest.raises(NotImplementedError):
            chaos.to_exception("openai")

    def test_str_representation(self) -> None:
        chaos = ContextLengthChaos(max_tokens=50000, on_call=1)
        assert str(chaos) == "llm_context_length(50000) on call 1"


class TestLLMChaosFactories:
    """Tests for LLM chaos factory functions."""

    def test_llm_rate_limit_factory(self) -> None:
        builder = llm_rate_limit(45.0)
        chaos = builder.on_call(2).build()
        assert isinstance(chaos, RateLimitChaos)
        assert chaos.retry_after == 45.0
        assert chaos.on_call == 2

    def test_llm_timeout_factory(self) -> None:
        builder = llm_timeout(60.0)
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, TimeoutChaos)

    def test_llm_server_error_factory(self) -> None:
        builder = llm_server_error("Service unavailable")
        chaos = builder.always().build()
        assert isinstance(chaos, ServerErrorChaos)
        assert chaos.message == "Service unavailable"
        assert chaos.always is True

    def test_llm_auth_error_factory(self) -> None:
        builder = llm_auth_error("Invalid token")
        chaos = builder.on_call(1).build()
        assert isinstance(chaos, AuthErrorChaos)
        assert chaos.message == "Invalid token"

    def test_llm_context_length_factory(self) -> None:
        builder = llm_context_length("Token limit exceeded")
        chaos = builder.after_calls(5).build()
        assert isinstance(chaos, ContextLengthChaos)
        assert chaos.message == "Token limit exceeded"
        assert chaos.after_calls == 5


class TestLLMChaosTurnTriggers:
    """Tests for turn-based triggers on LLM chaos."""

    def test_on_turn(self) -> None:
        chaos = RateLimitChaos(on_turn=2)
        assert not chaos.should_trigger(1, current_turn=1)
        assert chaos.should_trigger(1, current_turn=2)
        assert not chaos.should_trigger(1, current_turn=3)

    def test_after_turns(self) -> None:
        chaos = TimeoutChaos(after_turns=2)
        assert not chaos.should_trigger(1, completed_turns=1)
        # Triggers when completed_turns >= after_turns
        assert chaos.should_trigger(1, completed_turns=2)
        assert chaos.should_trigger(1, completed_turns=3)

    def test_between_turns(self) -> None:
        # between_turns is for between-turn operations (like history mutations)
        # It triggers when completed_turns >= after_turn AND current_turn == 0
        chaos = ServerErrorChaos(between_turns=(2, 4))
        # current_turn must be 0 (between turns, not during a turn)
        assert not chaos.should_trigger(1, current_turn=1, completed_turns=2)
        assert not chaos.should_trigger(1, current_turn=0, completed_turns=1)
        # Triggers when completed_turns >= 2 and current_turn == 0
        assert chaos.should_trigger(1, current_turn=0, completed_turns=2)
        assert chaos.should_trigger(1, current_turn=0, completed_turns=3)
