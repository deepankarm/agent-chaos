"""Unit tests for ChaosInjector - no API calls needed."""

from agent_chaos.core.injector import ChaosInjector
from agent_chaos.chaos.llm import RateLimitChaos, TimeoutChaos
from agent_chaos.chaos.stream import (
    SlowTTFTChaos,
    StreamCutChaos,
    StreamHangChaos,
)


def test_injector_on_call():
    """Test chaos triggering on specific call number."""
    chaos = RateLimitChaos(on_call=2)
    injector = ChaosInjector(chaos=[chaos])

    # Call 1: no chaos
    injector.increment_call()
    result = injector.next_llm_chaos("anthropic")
    assert result is None, "Call 1 should not trigger chaos"

    # Call 2: chaos triggers
    injector.increment_call()
    result = injector.next_llm_chaos("anthropic")
    assert result is not None, "Call 2 should trigger chaos"
    assert result.exception is not None, "Should have exception"

    # Call 3: chaos already used
    injector.increment_call()
    result = injector.next_llm_chaos("anthropic")
    assert result is None, "Call 3 should not trigger (already used)"

    print("✓ test_injector_on_call passed")


def test_injector_after_calls():
    """Test chaos triggering after N calls."""
    chaos = TimeoutChaos(after_calls=2)
    injector = ChaosInjector(chaos=[chaos])

    # Calls 1-2: no chaos
    injector.increment_call()
    assert injector.next_llm_chaos("anthropic") is None
    injector.increment_call()
    assert injector.next_llm_chaos("anthropic") is None

    # Call 3: chaos triggers (after 2 calls)
    injector.increment_call()
    result = injector.next_llm_chaos("anthropic")
    assert result is not None, "Call 3 should trigger chaos"

    print("✓ test_injector_after_calls passed")


def test_stream_chaos():
    """Test stream chaos detection methods."""
    injector = ChaosInjector(
        chaos=[
            SlowTTFTChaos(delay=1.5),
            StreamCutChaos(after_chunks=10),
            StreamHangChaos(after_chunks=20),
        ]
    )

    # TTFT delay
    assert injector.ttft_delay() == 1.5, "Should have 1.5s TTFT delay"

    # Stream cut
    assert not injector.should_cut(5), "Should not cut at chunk 5"
    assert injector.should_cut(10), "Should cut at chunk 10"
    assert injector.should_cut(15), "Should cut at chunk 15"

    # Stream hang
    assert not injector.should_hang(10), "Should not hang at chunk 10"
    assert injector.should_hang(20), "Should hang at chunk 20"

    print("✓ test_stream_chaos passed")


def test_provider_targeting():
    """Test provider-specific chaos targeting."""
    anthropic_chaos = RateLimitChaos(on_call=1, provider="anthropic")
    openai_chaos = TimeoutChaos(on_call=1, provider="openai")
    injector = ChaosInjector(chaos=[anthropic_chaos, openai_chaos])

    # Anthropic call should get anthropic chaos
    injector.increment_call()
    result = injector.next_llm_chaos("anthropic")
    assert result is not None, "Anthropic should get anthropic chaos"

    # Reset injector for next test
    injector = ChaosInjector(chaos=[anthropic_chaos, openai_chaos])

    # OpenAI call should get openai chaos
    injector.increment_call()
    result = injector.next_llm_chaos("openai")
    assert result is not None, "OpenAI should get openai chaos"

    print("✓ test_provider_targeting passed")


if __name__ == "__main__":
    print("🃏 agent-chaos — Unit Tests\n")
    print("=" * 60)

    test_injector_on_call()
    test_injector_after_calls()
    test_stream_chaos()
    test_provider_targeting()

    print("\n" + "=" * 60)
    print("✅ All unit tests passed!")
