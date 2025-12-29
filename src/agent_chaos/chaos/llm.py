"""LLM chaos types — raise exceptions at the LLM boundary."""

from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_chaos.chaos.base import ChaosPoint, ChaosResult, TriggerConfig
from agent_chaos.chaos.builder import ChaosBuilder


def _fake_response(status_code: int) -> httpx.Response:
    """Create a fake HTTP response for exception construction."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code=status_code, request=request)


@dataclass
class LLMChaos:
    """Base class for LLM chaos that raises exceptions."""

    message: str = ""
    on_call: int | None = None
    after_calls: int | None = None
    probability: float | None = None
    provider: str | None = None
    always: bool = False
    # Turn-based triggers
    on_turn: int | None = None
    after_turns: int | None = None
    between_turns: tuple[int, int] | None = None
    _trigger: TriggerConfig = field(init=False)

    def __post_init__(self):
        self._trigger = TriggerConfig(
            on_call=self.on_call,
            after_calls=self.after_calls,
            probability=self.probability,
            provider=self.provider,
            always=self.always,
            on_turn=self.on_turn,
            after_turns=self.after_turns,
            between_turns=self.between_turns,
        )

    @property
    def point(self) -> ChaosPoint:
        return ChaosPoint.LLM_CALL

    def should_trigger(self, call_number: int, **kwargs: Any) -> bool:
        provider = kwargs.get("provider")
        current_turn = kwargs.get("current_turn", 0)
        completed_turns = kwargs.get("completed_turns", 0)
        return self._trigger.should_trigger(
            call_number,
            provider=provider,
            current_turn=current_turn,
            completed_turns=completed_turns,
        )

    def apply(self, **kwargs: Any) -> ChaosResult:
        provider = kwargs.get("provider", "anthropic")
        exc = self.to_exception(provider)
        return ChaosResult.raise_exception(exc)

    def to_exception(self, provider: str) -> Exception:
        """Convert to provider-specific exception."""
        raise NotImplementedError


@dataclass
class RateLimitChaos(LLMChaos):
    """Simulates rate limit errors (429)."""

    retry_after: float = 30.0
    message: str = "Rate limit exceeded"

    def __str__(self) -> str:
        trigger = self._describe_trigger()
        return f"llm_rate_limit({self.retry_after}s){trigger}"

    def _describe_trigger(self) -> str:
        if self.on_call is not None:
            return f" on call {self.on_call}"
        if self.after_calls is not None:
            return f" after {self.after_calls} calls"
        if self.probability is not None and self.probability < 1.0:
            return f" @{int(self.probability * 100)}%"
        return ""

    def to_exception(self, provider: str) -> Exception:
        if provider == "anthropic":
            import anthropic

            return anthropic.RateLimitError(
                self.message,
                response=_fake_response(429),
                body={"error": {"type": "rate_limit_error", "message": self.message}},
            )
        raise NotImplementedError(f"Provider {provider} not implemented")


@dataclass
class TimeoutChaos(LLMChaos):
    """Simulates request timeout."""

    delay: float = 30.0
    message: str = "Request timed out"

    def __str__(self) -> str:
        trigger = self._describe_trigger()
        return f"llm_timeout({self.delay}s){trigger}"

    def _describe_trigger(self) -> str:
        if self.on_call is not None:
            return f" on call {self.on_call}"
        if self.after_calls is not None:
            return f" after {self.after_calls} calls"
        if self.probability is not None and self.probability < 1.0:
            return f" @{int(self.probability * 100)}%"
        return ""

    def to_exception(self, provider: str) -> Exception:
        if provider == "anthropic":
            import anthropic

            return anthropic.APITimeoutError(request=None)
        raise NotImplementedError(f"Provider {provider} not implemented")


@dataclass
class ServerErrorChaos(LLMChaos):
    """Simulates 500 internal server error."""

    status_code: int = 500
    message: str = "Internal server error"

    def __str__(self) -> str:
        trigger = self._describe_trigger()
        return f"llm_server_error({self.status_code}){trigger}"

    def _describe_trigger(self) -> str:
        if self.on_call is not None:
            return f" on call {self.on_call}"
        if self.after_calls is not None:
            return f" after {self.after_calls} calls"
        if self.probability is not None and self.probability < 1.0:
            return f" @{int(self.probability * 100)}%"
        return ""

    def to_exception(self, provider: str) -> Exception:
        if provider == "anthropic":
            import anthropic

            return anthropic.InternalServerError(
                self.message,
                response=_fake_response(self.status_code),
                body={
                    "error": {"type": "internal_server_error", "message": self.message}
                },
            )
        raise NotImplementedError(f"Provider {provider} not implemented")


@dataclass
class AuthErrorChaos(LLMChaos):
    """Simulates 401 authentication error."""

    message: str = "Invalid API key"

    def __str__(self) -> str:
        trigger = self._describe_trigger()
        return f"llm_auth_error{trigger}"

    def _describe_trigger(self) -> str:
        if self.on_call is not None:
            return f" on call {self.on_call}"
        if self.after_calls is not None:
            return f" after {self.after_calls} calls"
        if self.probability is not None and self.probability < 1.0:
            return f" @{int(self.probability * 100)}%"
        return ""

    def to_exception(self, provider: str) -> Exception:
        if provider == "anthropic":
            import anthropic

            return anthropic.AuthenticationError(
                self.message,
                response=_fake_response(401),
                body={
                    "error": {"type": "authentication_error", "message": self.message}
                },
            )
        raise NotImplementedError(f"Provider {provider} not implemented")


@dataclass
class ContextLengthChaos(LLMChaos):
    """Simulates context length exceeded error."""

    max_tokens: int = 200000
    message: str = "Context length exceeded"

    def __str__(self) -> str:
        trigger = self._describe_trigger()
        return f"llm_context_length({self.max_tokens}){trigger}"

    def _describe_trigger(self) -> str:
        if self.on_call is not None:
            return f" on call {self.on_call}"
        if self.after_calls is not None:
            return f" after {self.after_calls} calls"
        if self.probability is not None and self.probability < 1.0:
            return f" @{int(self.probability * 100)}%"
        return ""

    def to_exception(self, provider: str) -> Exception:
        if provider == "anthropic":
            import anthropic

            # BadRequestError in newer versions, InvalidRequestError in older
            exc_class = getattr(
                anthropic,
                "BadRequestError",
                getattr(anthropic, "InvalidRequestError", anthropic.APIError),
            )
            return exc_class(
                self.message,
                response=_fake_response(400),
                body={
                    "error": {"type": "invalid_request_error", "message": self.message}
                },
            )
        raise NotImplementedError(f"Provider {provider} not implemented")


# Factory functions


def llm_rate_limit(retry_after: float = 30.0) -> ChaosBuilder:
    """Create a rate limit chaos."""
    return ChaosBuilder(RateLimitChaos, retry_after=retry_after)


def llm_timeout(delay: float = 30.0) -> ChaosBuilder:
    """Create a timeout chaos."""
    return ChaosBuilder(TimeoutChaos, delay=delay)


def llm_server_error(message: str = "Internal server error") -> ChaosBuilder:
    """Create a server error chaos."""
    return ChaosBuilder(ServerErrorChaos, message=message)


def llm_auth_error(message: str = "Invalid API key") -> ChaosBuilder:
    """Create an auth error chaos."""
    return ChaosBuilder(AuthErrorChaos, message=message)


def llm_context_length(message: str = "Context length exceeded") -> ChaosBuilder:
    """Create a context length exceeded chaos."""
    return ChaosBuilder(ContextLengthChaos, message=message)
