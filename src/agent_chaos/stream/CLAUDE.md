# Stream Module

Stream wrappers that inject faults into LLM streaming responses.

## Provider Support

| Provider | Status | File |
|----------|--------|------|
| Anthropic | Implemented | `anthropic.py` |
| OpenAI | Stub only | `openai.py` |
| Gemini | Stub only | `gemini.py` |

## How It Works

When patched, `client.messages.stream()` returns a wrapped stream:

```
Original stream chunk → Chaos checks → (inject fault or pass through) → Consumer
```

## Fault Injection Points

Each chunk passes through these checks (in order):
1. **TTFT delay**: Delay first token arrival
2. **Stream hang**: Block indefinitely at chunk N
3. **Stream cut**: Raise connection error at chunk N
4. **Slow chunks**: Add delay between each chunk
5. **Corruption**: Mutate event data (wrong type, empty delta, truncate)

## Stream Wrappers (`anthropic.py`)

Sync wrappers:
- `ChaosAnthropicStream` - Context manager wrapper
- `ChaosMessageStream` - Iterator with fault injection
- `ChaosTextStream` - Text-only stream wrapper

Async wrappers:
- `ChaosAsyncAnthropicStream` - Async context manager
- `ChaosAsyncMessageStream` - Async iterator
- `ChaosAsyncTextStream` - Async text stream
- `ChaosAsyncStreamResponse` - AsyncStream from create with stream=True

## Base Mixin

`BaseStreamFaultMixin` provides shared fault injection logic for all wrappers.
