# Patch Module

Monkey-patching for LLM provider clients.

## Provider Support

| Provider | Status | File |
|----------|--------|------|
| Anthropic | Implemented | `providers/anthropic.py` |
| OpenAI | Stub only | `providers/openai.py` |
| Gemini | Stub only | `providers/gemini.py` |

## How It Works

1. **Discovery** (`discovery.py`): Finds installed LLM client modules
2. **Patcher** (`patcher.py`): Applies patches to client methods
3. **Providers** (`providers/`): Provider-specific patch implementations

## Patching Flow

```
Original: client.messages.create(...)
    ↓
Patched: chaos_wrapper(original_method, injector, recorder)
    ↓
1. Record call start (metrics.start_call)
2. Check chaos injection (injector.should_inject)
3. Call original (or inject fault)
4. Record tokens/tools
5. Return (possibly wrapped stream)
```

## Patched Methods (Anthropic)

- `messages.create()` - Non-streaming calls
- `messages.stream()` - Streaming calls
- Both sync and async variants

## Key Functions

```python
from agent_chaos.patch import patch_provider, unpatch_provider

# Apply patches
patch_provider("anthropic", injector, recorder)

# Remove patches
unpatch_provider("anthropic")
```
