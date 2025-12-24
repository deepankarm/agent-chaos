"""Core patcher for monkeypatching SDK methods."""

import json
import time
from functools import wraps
from typing import Any, Callable

from agent_chaos.core.injector import ChaosInjector
from agent_chaos.core.metrics import MetricsStore


class ChaosPatcher:
    """Patches SDK methods to inject chaos. Similar to OTel instrumentation."""

    def __init__(self, injector: ChaosInjector, metrics: MetricsStore):
        self.injector = injector
        self.metrics = metrics
        self._original_methods: dict[str, Callable] = {}
        self._patched = False

    def patch_all(self):
        """Patch all supported providers."""
        if self._patched:
            return
        self._patch_anthropic()
        self._patch_openai()
        self._patch_gemini()
        self._patched = True

    def unpatch_all(self):
        """Restore original methods."""
        for path, original in self._original_methods.items():
            self._set_method(path, original)
        self._original_methods.clear()
        self._patched = False

    def _save_original(self, path: str, method: Callable):
        """Save original method for restoration."""
        if path not in self._original_methods:
            self._original_methods[path] = method

    def _set_method(self, path: str, method: Callable):
        """Set a method on a module/class by dotted path."""
        parts = path.rsplit(".", 1)
        module = self._import_path(parts[0])
        setattr(module, parts[1], method)

    def _import_path(self, path: str):
        """Import and return object at dotted path."""
        parts = path.split(".")
        obj = __import__(parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

    def _patch_anthropic(self):
        """Patch anthropic sync and async messages.create and .stream"""
        try:
            from anthropic.resources import AsyncMessages, Messages
            from anthropic.resources.beta.messages import (
                AsyncMessages as BetaAsyncMessages,
            )
            from anthropic.resources.beta.messages import (
                Messages as BetaMessages,
            )
        except ImportError:
            return

        # Patch all message classes
        self._patch_sync_messages(Messages, "anthropic.resources.Messages")
        self._patch_async_messages(AsyncMessages, "anthropic.resources.AsyncMessages")
        self._patch_sync_messages(
            BetaMessages, "anthropic.resources.beta.messages.Messages"
        )
        self._patch_beta_async_messages(
            BetaAsyncMessages, "anthropic.resources.beta.messages.AsyncMessages"
        )

    def _patch_sync_messages(self, messages_cls: type, path_prefix: str):
        """Patch sync Messages class (create and stream)."""
        injector = self.injector
        metrics = self.metrics

        # Patch create
        self._save_original(f"{path_prefix}.create", messages_cls.create)
        original_create = messages_cls.create

        @wraps(original_create)
        def patched_create(self_msg, **kwargs):
            return _execute_with_chaos_sync(
                lambda kw: original_create(self_msg, **kw),
                injector,
                metrics,
                kwargs,
            )

        messages_cls.create = patched_create

        # Patch stream if exists
        if hasattr(messages_cls, "stream"):
            self._save_original(f"{path_prefix}.stream", messages_cls.stream)
            original_stream = messages_cls.stream

            @wraps(original_stream)
            def patched_stream(self_msg, **kwargs):
                call_id = metrics.start_call("anthropic")
                injector.increment_call()

                kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
                if chaos_result := injector.next_llm_chaos("anthropic"):
                    if chaos_result.exception:
                        metrics.record_fault(
                            call_id, chaos_result.exception, provider="anthropic"
                        )
                        metrics.end_call(
                            call_id, success=False, error=chaos_result.exception
                        )
                        raise chaos_result.exception

                from agent_chaos.stream.anthropic import ChaosAnthropicStream

                return ChaosAnthropicStream(
                    original_stream(self_msg, **kwargs), injector, metrics, call_id
                )

            messages_cls.stream = patched_stream

    def _patch_async_messages(self, messages_cls: type, path_prefix: str):
        """Patch async Messages class (create only)."""
        injector = self.injector
        metrics = self.metrics

        self._save_original(f"{path_prefix}.create", messages_cls.create)
        original_create = messages_cls.create

        @wraps(original_create)
        async def patched_create(self_msg, **kwargs):
            return await _execute_with_chaos_async(
                lambda kw: original_create(self_msg, **kw),
                injector,
                metrics,
                kwargs,
            )

        messages_cls.create = patched_create

    def _patch_beta_async_messages(self, messages_cls: type, path_prefix: str):
        """Patch beta async Messages class with streaming support."""
        injector = self.injector
        metrics = self.metrics

        # Patch create (handles stream=True)
        self._save_original(f"{path_prefix}.create", messages_cls.create)
        original_create = messages_cls.create

        @wraps(original_create)
        async def patched_create(self_msg, **kwargs):
            call_id = metrics.start_call("anthropic")
            injector.increment_call()
            is_streaming = kwargs.get("stream", False)

            kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
            _maybe_record_anthropic_tool_results_in_request(
                metrics, kwargs, current_call_id=call_id
            )
            if chaos_result := injector.next_llm_chaos("anthropic"):
                if chaos_result.exception:
                    metrics.record_fault(
                        call_id, chaos_result.exception, provider="anthropic"
                    )
                    metrics.end_call(
                        call_id, success=False, error=chaos_result.exception
                    )
                    raise chaos_result.exception

            start = time.monotonic()
            try:
                response = await original_create(self_msg, **kwargs)

                if is_streaming:
                    from agent_chaos.stream.anthropic import ChaosAsyncStreamResponse

                    return ChaosAsyncStreamResponse(
                        response, injector, metrics, call_id
                    )

                metrics.record_latency(call_id, time.monotonic() - start)
                _maybe_record_anthropic_response_metadata(metrics, call_id, response)
                metrics.end_call(call_id, success=True)
                return response
            except Exception as e:
                metrics.end_call(call_id, success=False, error=e)
                raise

        messages_cls.create = patched_create

        # Patch stream if exists
        if hasattr(messages_cls, "stream"):
            self._save_original(f"{path_prefix}.stream", messages_cls.stream)
            original_stream = messages_cls.stream

            @wraps(original_stream)
            def patched_stream(self_msg, **kwargs):
                call_id = metrics.start_call("anthropic")
                injector.increment_call()

                kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
                if chaos_result := injector.next_llm_chaos("anthropic"):
                    if chaos_result.exception:
                        metrics.record_fault(
                            call_id, chaos_result.exception, provider="anthropic"
                        )
                        metrics.end_call(
                            call_id, success=False, error=chaos_result.exception
                        )
                        raise chaos_result.exception

                from agent_chaos.stream.anthropic import ChaosAsyncAnthropicStream

                return ChaosAsyncAnthropicStream(
                    original_stream(self_msg, **kwargs), injector, metrics, call_id
                )

            messages_cls.stream = patched_stream

    def _patch_openai(self):
        """Patch openai.OpenAI.chat.completions.create"""
        raise NotImplementedError("not implemented")

    def _patch_gemini(self):
        """Patch google.generativeai.GenerativeModel.generate_content"""
        raise NotImplementedError("not implemented")


def _maybe_mutate_tools(
    kwargs: dict, injector: ChaosInjector, metrics: MetricsStore
) -> dict:
    """Mutate tool results if configured."""
    if not injector.should_mutate_tools():
        return kwargs
    return _mutate_anthropic_tool_results(kwargs, injector, metrics)


def _maybe_record_anthropic_response_metadata(
    metrics: MetricsStore, call_id: str, response: Any
) -> None:
    """Best-effort extraction of usage + tool_use blocks from an Anthropic response."""
    # Token usage
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", None)
            out_tok = getattr(usage, "output_tokens", None)
            total = getattr(usage, "total_tokens", None)
            model = getattr(response, "model", None) or getattr(
                response, "model_name", None
            )
            if any(v is not None for v in [in_tok, out_tok, total, model]):
                metrics.record_token_usage(
                    call_id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=total,
                    model=model,
                    provider="anthropic",
                )
    except Exception:
        pass

    # Tool-use blocks (LLM requested tools)
    try:
        content = getattr(response, "content", None)
        if isinstance(content, list):
            for block in content:
                b_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if b_type != "tool_use":
                    continue
                tool_name = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, dict) else None
                )
                tool_use_id = getattr(block, "id", None) or (
                    block.get("id") if isinstance(block, dict) else None
                )
                tool_input = getattr(block, "input", None) or (
                    block.get("input") if isinstance(block, dict) else None
                )
                input_bytes = None
                try:
                    if tool_input is not None:
                        input_bytes = len(
                            json.dumps(tool_input, ensure_ascii=False).encode("utf-8")
                        )
                except Exception:
                    input_bytes = None
                if tool_name:
                    metrics.record_tool_use(
                        call_id,
                        tool_name=str(tool_name),
                        tool_use_id=str(tool_use_id) if tool_use_id else None,
                        input_bytes=input_bytes,
                        provider="anthropic",
                    )
                    # Non-intrusive inference: treat tool_use completion as tool_start.
                    if tool_use_id:
                        metrics.record_tool_start(
                            tool_name=str(tool_name),
                            tool_use_id=str(tool_use_id),
                            call_id=call_id,
                            input_bytes=input_bytes,
                            provider="anthropic",
                        )
    except Exception:
        pass


def _maybe_record_anthropic_tool_results_in_request(
    metrics: MetricsStore, mutated_kwargs: dict, *, current_call_id: str
) -> None:
    """Infer tool_end when we see tool_result blocks in a subsequent LLM request."""
    try:
        messages = mutated_kwargs.get("messages", []) or []
        for msg in messages:
            if not (isinstance(msg, dict) and msg.get("role") == "user"):
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                tool_use_id = block.get("tool_use_id")
                if not tool_use_id:
                    continue
                is_error = bool(block.get("is_error")) if "is_error" in block else None
                out_bytes = None
                try:
                    out_bytes = len(
                        json.dumps(block.get("content", ""), ensure_ascii=False).encode(
                            "utf-8"
                        )
                    )
                except Exception:
                    out_bytes = None
                metrics.record_tool_result_seen(
                    tool_use_id=str(tool_use_id),
                    is_error=is_error,
                    output_bytes=out_bytes,
                    resolved_in_call_id=current_call_id,
                    provider="anthropic",
                )
    except Exception:
        pass


def _execute_with_chaos_sync(
    execute_fn: Callable[[dict], Any],
    injector: ChaosInjector,
    metrics: MetricsStore,
    kwargs: dict,
) -> Any:
    """Execute sync call with chaos injection."""
    call_id = metrics.start_call("anthropic")
    injector.increment_call()

    mutated_kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
    _maybe_record_anthropic_tool_results_in_request(
        metrics, mutated_kwargs, current_call_id=call_id
    )

    if chaos_result := injector.next_llm_chaos("anthropic"):
        if chaos_result.exception:
            metrics.record_fault(call_id, chaos_result.exception, provider="anthropic")
            metrics.end_call(call_id, success=False, error=chaos_result.exception)
            raise chaos_result.exception

    start = time.monotonic()
    try:
        response = execute_fn(mutated_kwargs)
        metrics.record_latency(call_id, time.monotonic() - start)
        _maybe_record_anthropic_response_metadata(metrics, call_id, response)
        metrics.end_call(call_id, success=True)
        return response
    except Exception as e:
        metrics.end_call(call_id, success=False, error=e)
        raise


async def _execute_with_chaos_async(
    execute_fn: Callable[[dict], Any],
    injector: ChaosInjector,
    metrics: MetricsStore,
    kwargs: dict,
) -> Any:
    """Execute async call with chaos injection."""
    call_id = metrics.start_call("anthropic")
    injector.increment_call()

    mutated_kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
    _maybe_record_anthropic_tool_results_in_request(
        metrics, mutated_kwargs, current_call_id=call_id
    )

    if chaos_result := injector.next_llm_chaos("anthropic"):
        if chaos_result.exception:
            metrics.record_fault(call_id, chaos_result.exception, provider="anthropic")
            metrics.end_call(call_id, success=False, error=chaos_result.exception)
            raise chaos_result.exception

    start = time.monotonic()
    try:
        response = await execute_fn(mutated_kwargs)
        metrics.record_latency(call_id, time.monotonic() - start)
        _maybe_record_anthropic_response_metadata(metrics, call_id, response)
        metrics.end_call(call_id, success=True)
        return response
    except Exception as e:
        metrics.end_call(call_id, success=False, error=e)
        raise


def _mutate_anthropic_tool_results(
    kwargs: dict, injector: ChaosInjector, metrics: MetricsStore
) -> dict:
    """Mutate tool_result blocks in messages using new chaos system."""
    messages = kwargs.get("messages", [])
    mutated_messages = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                mutated_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_name = block.get("tool_use_id", "unknown")
                        original_result = block.get("content", "")
                        if isinstance(original_result, list):
                            original_result = json.dumps(original_result)
                        elif not isinstance(original_result, str):
                            original_result = str(original_result)

                        if chaos_result := injector.next_tool_chaos(
                            tool_name, original_result
                        ):
                            if chaos_result.mutated is not None:
                                block = {**block, "content": chaos_result.mutated}
                                metrics.record_fault(
                                    "tool_mutation",
                                    f"mutated {tool_name}",
                                    provider="anthropic",
                                )
                    mutated_content.append(block)
                msg = {**msg, "content": mutated_content}
        mutated_messages.append(msg)

    return {**kwargs, "messages": mutated_messages}
