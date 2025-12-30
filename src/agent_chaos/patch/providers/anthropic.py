"""Anthropic provider patcher."""

from __future__ import annotations

import json
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from agent_chaos.patch.base import BaseProviderPatcher

if TYPE_CHECKING:
    from agent_chaos.core.injector import ChaosInjector
    from agent_chaos.core.metrics import MetricsStore


class AnthropicPatcher(BaseProviderPatcher):
    """Patches Anthropic SDK methods to inject chaos."""

    provider_name = "anthropic"

    def __init__(self):
        super().__init__()
        self._injector: ChaosInjector | None = None
        self._metrics: MetricsStore | None = None

    def patch(self, injector: "ChaosInjector", metrics: "MetricsStore") -> None:
        """Apply patches to Anthropic SDK."""
        if self._patched:
            return

        self._injector = injector
        self._metrics = metrics

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
        self._patched = True

    def _patch_sync_messages(self, messages_cls: type, path_prefix: str):
        """Patch sync Messages class (create and stream)."""
        injector = self._injector
        metrics = self._metrics

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
                call_number = injector.increment_call()

                # Apply user input chaos on FIRST call only
                if call_number == 1:
                    messages = kwargs.get("messages", [])
                    if messages:
                        mutated_messages, original_input = (
                            _apply_user_chaos_to_messages(messages, injector, metrics)
                        )
                        if original_input is not None and injector._ctx is not None:
                            injector._ctx.agent_input = original_input
                        kwargs = {**kwargs, "messages": mutated_messages}

                # Apply context and tool mutations
                kwargs = _maybe_mutate_context(kwargs, injector, metrics)
                kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
                if chaos_result := injector.next_llm_chaos("anthropic"):
                    if chaos_result.exception:
                        metrics.record_fault(
                            call_id,
                            chaos_result.exception,
                            provider="anthropic",
                            chaos_point="LLM",
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
        injector = self._injector
        metrics = self._metrics

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
        injector = self._injector
        metrics = self._metrics

        # Patch create (handles stream=True)
        self._save_original(f"{path_prefix}.create", messages_cls.create)
        original_create = messages_cls.create

        @wraps(original_create)
        async def patched_create(self_msg, **kwargs):
            call_id = metrics.start_call("anthropic")
            call_number = injector.increment_call()
            is_streaming = kwargs.get("stream", False)

            # Apply user input chaos on FIRST call only
            if call_number == 1:
                messages = kwargs.get("messages", [])
                if messages:
                    mutated_messages, original_input = _apply_user_chaos_to_messages(
                        messages, injector, metrics
                    )
                    if original_input is not None and injector._ctx is not None:
                        injector._ctx.agent_input = original_input
                    kwargs = {**kwargs, "messages": mutated_messages}

            kwargs = _maybe_mutate_context(kwargs, injector, metrics)
            kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
            _maybe_record_anthropic_tool_results_in_request(
                metrics, kwargs, current_call_id=call_id
            )
            if chaos_result := injector.next_llm_chaos("anthropic"):
                if chaos_result.exception:
                    metrics.record_fault(
                        call_id,
                        chaos_result.exception,
                        provider="anthropic",
                        chaos_point="LLM",
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
                call_number = injector.increment_call()

                # Apply user input chaos on FIRST call only
                if call_number == 1:
                    messages = kwargs.get("messages", [])
                    if messages:
                        mutated_messages, original_input = (
                            _apply_user_chaos_to_messages(messages, injector, metrics)
                        )
                        if original_input is not None and injector._ctx is not None:
                            injector._ctx.agent_input = original_input
                        kwargs = {**kwargs, "messages": mutated_messages}

                # Apply context and tool mutations
                kwargs = _maybe_mutate_context(kwargs, injector, metrics)
                kwargs = _maybe_mutate_tools(kwargs, injector, metrics)
                if chaos_result := injector.next_llm_chaos("anthropic"):
                    if chaos_result.exception:
                        metrics.record_fault(
                            call_id,
                            chaos_result.exception,
                            provider="anthropic",
                            chaos_point="LLM",
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


# -----------------------------------------------------------------------------
# Helper functions (Anthropic-specific)
# -----------------------------------------------------------------------------


def _maybe_mutate_tools(
    kwargs: dict, injector: "ChaosInjector", metrics: "MetricsStore"
) -> dict:
    """Mutate tool results if configured."""
    if not injector.should_mutate_tools():
        return kwargs
    return _mutate_anthropic_tool_results(kwargs, injector, metrics)


def _maybe_record_anthropic_response_metadata(
    metrics: "MetricsStore", call_id: str, response: Any
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
                tool_args = None
                try:
                    if tool_input is not None:
                        input_bytes = len(
                            json.dumps(tool_input, ensure_ascii=False).encode("utf-8")
                        )
                        # Capture args for conversation view
                        if isinstance(tool_input, dict):
                            tool_args = tool_input
                except Exception:
                    input_bytes = None
                if tool_name:
                    metrics.record_tool_use(
                        call_id,
                        tool_name=str(tool_name),
                        tool_use_id=str(tool_use_id) if tool_use_id else None,
                        input_bytes=input_bytes,
                        tool_args=tool_args,
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
    metrics: "MetricsStore", mutated_kwargs: dict, *, current_call_id: str
) -> None:
    """Infer tool_end when we see tool_result blocks in a subsequent LLM request."""
    try:
        messages = mutated_kwargs.get("messages", []) or []

        # First pass: extract tool_use_id -> tool_name mapping from assistant messages
        # This ensures tool names are available even in streaming scenarios
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input")
                            if tool_id and tool_name:
                                # Populate the mapping for later lookup
                                if tool_id not in metrics._tool_use_to_tool_name:
                                    metrics._tool_use_to_tool_name[tool_id] = tool_name
                                    metrics._tool_use_to_call_id[tool_id] = (
                                        current_call_id
                                    )
                                # Also add to conversation if not already there
                                if tool_id not in metrics._tool_use_in_conversation:
                                    metrics._tool_use_in_conversation.add(tool_id)
                                    metrics.add_conversation_entry(
                                        "tool_call",
                                        tool_name=tool_name,
                                        tool_use_id=tool_id,
                                        args=(
                                            tool_input
                                            if isinstance(tool_input, dict)
                                            else None
                                        ),
                                    )

        # Second pass: process tool_result blocks
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
                result_content = None
                try:
                    raw_content = block.get("content", "")
                    # Capture result content for conversation view
                    if isinstance(raw_content, str):
                        result_content = raw_content
                    elif isinstance(raw_content, list):
                        # Handle list of content blocks (extract text)
                        texts = []
                        for item in raw_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                texts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                texts.append(item)
                        result_content = (
                            " ".join(texts) if texts else json.dumps(raw_content)
                        )
                    else:
                        result_content = (
                            json.dumps(raw_content) if raw_content else None
                        )
                    out_bytes = len(
                        json.dumps(raw_content, ensure_ascii=False).encode("utf-8")
                    )
                except Exception:
                    out_bytes = None
                metrics.record_tool_result_seen(
                    tool_use_id=str(tool_use_id),
                    is_error=is_error,
                    output_bytes=out_bytes,
                    result=result_content,
                    resolved_in_call_id=current_call_id,
                    provider="anthropic",
                )
    except Exception:
        pass


def _extract_last_user_text(messages: list[dict]) -> tuple[int, int | None, str | None]:
    """Extract the last user text message from messages array.

    Returns:
        (message_index, content_block_index, text_content)
        - message_index: Index of the message containing user text
        - content_block_index: Index within content list (None if content is string)
        - text_content: The actual text content

    Handles both formats:
        - {"role": "user", "content": "Hello"}  # String content
        - {"role": "user", "content": [{"type": "text", "text": "Hello"}]}  # Block content
    """
    # Search from end to find last user message with text
    for msg_idx in range(len(messages) - 1, -1, -1):
        msg = messages[msg_idx]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        content = msg.get("content")

        # String content format
        if isinstance(content, str):
            return (msg_idx, None, content)

        # List/block content format (Anthropic style)
        if isinstance(content, list):
            # Find last text block (not tool_result)
            for block_idx in range(len(content) - 1, -1, -1):
                block = content[block_idx]
                if isinstance(block, dict) and block.get("type") == "text":
                    return (msg_idx, block_idx, block.get("text", ""))
                # Handle plain string in list
                if isinstance(block, str):
                    return (msg_idx, block_idx, block)

    return (-1, None, None)


def _apply_user_chaos_to_messages(
    messages: list[dict],
    injector: "ChaosInjector",
    metrics: "MetricsStore",
) -> tuple[list[dict], str | None]:
    """Apply user input chaos to the messages array.

    This modifies the last user text message if user chaos is configured.
    Only called on the FIRST LLM call.

    Returns:
        (mutated_messages, original_user_input)
    """
    # Extract user input first
    msg_idx, block_idx, original_text = _extract_last_user_text(messages)
    if original_text is None:
        return messages, None

    # Always record the original user message to conversation (for UI display)
    # Deduplication is handled automatically in add_conversation_entry
    metrics.add_conversation_entry("user", content=original_text)

    # If no chaos configured, return early
    if not injector.has_user_chaos():
        return messages, original_text

    # Apply user chaos
    mutated_text, chaos_obj = injector.apply_user_chaos(original_text)

    if chaos_obj is None or mutated_text == original_text:
        return messages, original_text

    # Create mutated messages (deep copy the affected message)
    mutated_messages = list(messages)
    original_msg = messages[msg_idx]

    if block_idx is None:
        # String content format
        mutated_messages[msg_idx] = {**original_msg, "content": mutated_text}
    else:
        # Block content format
        content = list(original_msg.get("content", []))
        block = content[block_idx]
        if isinstance(block, dict):
            content[block_idx] = {**block, "text": mutated_text}
        else:
            content[block_idx] = mutated_text
        mutated_messages[msg_idx] = {**original_msg, "content": content}

    # Record the chaos injection
    chaos_fn_name = None
    chaos_fn_doc = None
    if hasattr(chaos_obj, "mutator") and chaos_obj.mutator:
        chaos_fn_name = getattr(chaos_obj.mutator, "__name__", None)
        doc = getattr(chaos_obj.mutator, "__doc__", None)
        if doc:
            chaos_fn_doc = doc.strip().split("\n")[0]

    metrics.record_fault(
        "user_input_mutation",
        chaos_obj,
        provider="",
        chaos_point="USER_INPUT",
        chaos_fn_name=chaos_fn_name,
        chaos_fn_doc=chaos_fn_doc,
        original=original_text,
        mutated=mutated_text,
    )

    return mutated_messages, original_text


def _compute_context_diff(original: list[dict], mutated: list[dict]) -> dict[str, Any]:
    """Compute what changed between original and mutated message lists.

    Returns a dict with:
    - summary: "3 -> 4 messages"
    - added: list of added messages (with content)
    - removed: list of removed messages (with content)
    - modified_indices: indices of messages that were modified
    """
    original_count = len(original)
    mutated_count = len(mutated)

    # Find added messages (messages in mutated but not in original)
    # We compare by serializing to detect exact matches
    def msg_key(m: dict) -> str:
        return json.dumps(m, sort_keys=True, default=str)

    original_set = {msg_key(m) for m in original}
    mutated_set = {msg_key(m) for m in mutated}

    added = [m for m in mutated if msg_key(m) not in original_set]
    removed = [m for m in original if msg_key(m) not in mutated_set]

    # Extract content for display
    def extract_content(msg: dict) -> str:
        """Extract displayable content from a message."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content[:500]  # Truncate for storage
        if isinstance(content, list):
            # Handle content blocks (Anthropic format)
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", "")[:200])
                    elif block.get("type") == "tool_result":
                        texts.append(f"[tool_result: {block.get('tool_use_id', '')}]")
                elif isinstance(block, str):
                    texts.append(block[:200])
            return " | ".join(texts)[:500]
        return str(content)[:500]

    added_content = [
        {"role": m.get("role", "unknown"), "content": extract_content(m)} for m in added
    ]
    removed_content = [
        {"role": m.get("role", "unknown"), "content": extract_content(m)}
        for m in removed
    ]

    return {
        "summary": f"{original_count} -> {mutated_count} messages",
        "added": added_content,
        "removed": removed_content,
        "added_count": len(added),
        "removed_count": len(removed),
    }


def _maybe_mutate_context(
    kwargs: dict, injector: "ChaosInjector", metrics: "MetricsStore"
) -> dict:
    """Apply context chaos to mutate the messages array."""
    messages = kwargs.get("messages", [])
    if not messages:
        return kwargs

    if result := injector.next_context_chaos(messages):
        chaos_result, chaos_obj = result
        if chaos_result.mutated is not None:
            # Extract function metadata for UI
            chaos_fn_name = None
            chaos_fn_doc = None
            if hasattr(chaos_obj, "mutator") and chaos_obj.mutator:
                chaos_fn_name = getattr(chaos_obj.mutator, "__name__", None)
                doc = getattr(chaos_obj.mutator, "__doc__", None)
                if doc:
                    # Get first line of docstring
                    chaos_fn_doc = doc.strip().split("\n")[0]

            # Compute detailed diff of what changed
            diff = _compute_context_diff(messages, chaos_result.mutated)

            metrics.record_fault(
                "context_mutation",
                chaos_obj,
                provider="anthropic",
                chaos_point="CONTEXT",
                chaos_fn_name=chaos_fn_name,
                chaos_fn_doc=chaos_fn_doc,
                original=diff["summary"],
                # New fields for detailed mutation info
                added_messages=diff["added"],
                removed_messages=diff["removed"],
                added_count=diff["added_count"],
                removed_count=diff["removed_count"],
            )
            return {**kwargs, "messages": chaos_result.mutated}

    return kwargs


def _execute_with_chaos_sync(
    execute_fn: Callable[[dict], Any],
    injector: "ChaosInjector",
    metrics: "MetricsStore",
    kwargs: dict,
) -> Any:
    """Execute sync call with chaos injection."""
    call_id = metrics.start_call("anthropic")
    call_number = injector.increment_call()

    mutated_kwargs = kwargs

    # Apply user input chaos on FIRST call only
    if call_number == 1:
        messages = mutated_kwargs.get("messages", [])
        if messages:
            mutated_messages, original_input = _apply_user_chaos_to_messages(
                messages, injector, metrics
            )
            if original_input is not None:
                # Store original user input in context
                if injector._ctx is not None:
                    injector._ctx.agent_input = original_input
            mutated_kwargs = {**mutated_kwargs, "messages": mutated_messages}

    # Apply context mutation (messages array)
    mutated_kwargs = _maybe_mutate_context(mutated_kwargs, injector, metrics)
    # Then apply tool result mutations
    mutated_kwargs = _maybe_mutate_tools(mutated_kwargs, injector, metrics)
    _maybe_record_anthropic_tool_results_in_request(
        metrics, mutated_kwargs, current_call_id=call_id
    )

    if chaos_result := injector.next_llm_chaos("anthropic"):
        if chaos_result.exception:
            metrics.record_fault(
                call_id,
                chaos_result.exception,
                provider="anthropic",
                chaos_point="LLM",
            )
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
    injector: "ChaosInjector",
    metrics: "MetricsStore",
    kwargs: dict,
) -> Any:
    """Execute async call with chaos injection."""
    call_id = metrics.start_call("anthropic")
    call_number = injector.increment_call()

    mutated_kwargs = kwargs

    # Apply user input chaos on FIRST call only
    if call_number == 1:
        messages = mutated_kwargs.get("messages", [])
        if messages:
            mutated_messages, original_input = _apply_user_chaos_to_messages(
                messages, injector, metrics
            )
            if original_input is not None:
                # Store original user input in context
                if injector._ctx is not None:
                    injector._ctx.agent_input = original_input
            mutated_kwargs = {**mutated_kwargs, "messages": mutated_messages}

    # Apply context mutation (messages array)
    mutated_kwargs = _maybe_mutate_context(mutated_kwargs, injector, metrics)
    # Then apply tool result mutations
    mutated_kwargs = _maybe_mutate_tools(mutated_kwargs, injector, metrics)
    _maybe_record_anthropic_tool_results_in_request(
        metrics, mutated_kwargs, current_call_id=call_id
    )

    if chaos_result := injector.next_llm_chaos("anthropic"):
        if chaos_result.exception:
            metrics.record_fault(
                call_id,
                chaos_result.exception,
                provider="anthropic",
                chaos_point="LLM",
            )
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
    kwargs: dict, injector: "ChaosInjector", metrics: "MetricsStore"
) -> dict:
    """Mutate tool_result blocks in messages using new chaos system."""
    messages = kwargs.get("messages", [])

    # Build tool_use_id -> tool_name mapping from assistant messages
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "unknown")
                        if tool_id:
                            tool_id_to_name[tool_id] = tool_name

    mutated_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                mutated_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        # Look up the actual tool name from the mapping
                        tool_name = tool_id_to_name.get(tool_use_id, tool_use_id)
                        original_result = block.get("content", "")
                        if isinstance(original_result, list):
                            original_result = json.dumps(original_result)
                        elif not isinstance(original_result, str):
                            original_result = str(original_result)

                        if result := injector.next_tool_chaos(
                            tool_name, original_result
                        ):
                            chaos_result, chaos_obj = result
                            if chaos_result.mutated is not None:
                                block = {**block, "content": chaos_result.mutated}
                                # Extract function metadata for UI
                                chaos_fn_name = None
                                chaos_fn_doc = None
                                if hasattr(chaos_obj, "mutator") and chaos_obj.mutator:
                                    chaos_fn_name = getattr(
                                        chaos_obj.mutator, "__name__", None
                                    )
                                    doc = getattr(chaos_obj.mutator, "__doc__", None)
                                    if doc:
                                        # Get first line of docstring
                                        chaos_fn_doc = doc.strip().split("\n")[0]
                                metrics.record_fault(
                                    "tool_mutation",
                                    chaos_obj,
                                    provider="anthropic",
                                    chaos_point="TOOL",
                                    chaos_fn_name=chaos_fn_name,
                                    chaos_fn_doc=chaos_fn_doc,
                                    target_tool=tool_name,
                                    original=original_result,
                                    mutated=chaos_result.mutated,
                                )
                    mutated_content.append(block)
                msg = {**msg, "content": mutated_content}
        mutated_messages.append(msg)

    return {**kwargs, "messages": mutated_messages}
