"""Chaos fuzzing — generate random chaos configurations.

This module provides utilities to generate randomized chaos scenarios
for exploring the failure space of AI agents.

Example:
    from agent_chaos import Scenario, Turn, fuzz_chaos, ChaosSpace

    baseline = Scenario(
        name="order-check",
        agent=my_agent,
        turns=[Turn("Check order #123")],
    )

    # Generate 20 random chaos variations
    fuzzed = fuzz_chaos(baseline, n=20, seed=42)

    def get_scenarios():
        return [baseline] + fuzzed
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agent_chaos.chaos.builder import ChaosBuilder
from agent_chaos.chaos.llm import (
    RateLimitChaos,
    TimeoutChaos,
    ServerErrorChaos,
    AuthErrorChaos,
    ContextLengthChaos,
)
from agent_chaos.chaos.stream import (
    StreamCutChaos,
    StreamHangChaos,
    SlowTTFTChaos,
    SlowChunksChaos,
)
from agent_chaos.chaos.tool import (
    ToolErrorChaos,
    ToolEmptyChaos,
    ToolTimeoutChaos,
    ToolMutateChaos,
)
from agent_chaos.chaos.context import ContextMutateChaos

if TYPE_CHECKING:
    from agent_chaos.scenario.model import Scenario
    from agent_chaos.chaos.base import Chaos
    from agent_chaos.core.context import ChaosContext


# Type aliases for user-provided mutators
ToolMutator = Callable[[str, str], str]  # (tool_name, result) -> result
ToolMutatorWithCtx = Callable[["ChaosContext", str, str], str]
ContextMutator = Callable[[list], list]  # (messages) -> messages
ContextMutatorWithCtx = Callable[["ChaosContext", list], list]


# LLM fault type to class mapping
_LLM_FAULTS: dict[str, type] = {
    "rate_limit": RateLimitChaos,
    "timeout": TimeoutChaos,
    "server_error": ServerErrorChaos,
    "auth_error": AuthErrorChaos,
    "context_length": ContextLengthChaos,
}

# Stream fault type to class mapping
_STREAM_FAULTS: dict[str, type] = {
    "stream_cut": StreamCutChaos,
    "stream_hang": StreamHangChaos,
    "slow_ttft": SlowTTFTChaos,
    "slow_chunks": SlowChunksChaos,
}

# Tool fault type to class mapping
_TOOL_FAULTS: dict[str, type] = {
    "error": ToolErrorChaos,
    "timeout": ToolTimeoutChaos,
    "empty": ToolEmptyChaos,
}


# =============================================================================
# Nested Configuration Dataclasses
# =============================================================================


@dataclass
class LLMFuzzConfig:
    """Configuration for LLM-related chaos fuzzing.

    Attributes:
        enabled: Whether to include LLM chaos in fuzzing.
        probability: Probability weight for selecting LLM chaos (0.0-1.0).
        faults: List of fault types to choose from.
            Options: "rate_limit", "timeout", "server_error", "auth_error", "context_length"
        retry_after_range: Range for retry_after parameter in rate_limit (seconds).
        timeout_range: Range for delay parameter in timeout (seconds).

    Example:
        llm = LLMFuzzConfig(
            probability=0.5,
            faults=["rate_limit", "timeout"],
        )
    """

    enabled: bool = True
    probability: float = 0.3
    faults: list[str] = field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error"]
    )
    retry_after_range: tuple[float, float] = (1.0, 30.0)
    timeout_range: tuple[float, float] = (5.0, 60.0)

    @classmethod
    def disabled(cls) -> "LLMFuzzConfig":
        """Create a disabled LLM fuzz config."""
        return cls(enabled=False)

    @classmethod
    def heavy(cls) -> "LLMFuzzConfig":
        """Create a heavy LLM fuzz config for stress testing."""
        return cls(
            probability=0.6,
            faults=["rate_limit", "timeout", "server_error", "context_length"],
        )


@dataclass
class StreamFuzzConfig:
    """Configuration for stream-related chaos fuzzing.

    Attributes:
        enabled: Whether to include stream chaos in fuzzing.
        probability: Probability weight for selecting stream chaos (0.0-1.0).
        faults: List of fault types to choose from.
            Options: "stream_cut", "stream_hang", "slow_ttft", "slow_chunks"
        cut_after_chunks: Range for after_chunks parameter in stream_cut/hang.
        ttft_delay_range: Range for delay in slow_ttft (seconds).
        chunk_delay_range: Range for delay in slow_chunks (seconds).

    Example:
        stream = StreamFuzzConfig(
            probability=0.4,
            faults=["stream_cut", "slow_ttft"],
        )
    """

    enabled: bool = True
    probability: float = 0.2
    faults: list[str] = field(default_factory=lambda: ["stream_cut", "slow_ttft"])
    cut_after_chunks: tuple[int, int] = (3, 20)
    ttft_delay_range: tuple[float, float] = (1.0, 5.0)
    chunk_delay_range: tuple[float, float] = (0.1, 0.5)

    @classmethod
    def disabled(cls) -> "StreamFuzzConfig":
        """Create a disabled stream fuzz config."""
        return cls(enabled=False)

    @classmethod
    def heavy(cls) -> "StreamFuzzConfig":
        """Create a heavy stream fuzz config for stress testing."""
        return cls(
            probability=0.5,
            faults=["stream_cut", "stream_hang", "slow_ttft", "slow_chunks"],
        )


@dataclass
class ToolFuzzConfig:
    """Configuration for tool-related chaos fuzzing.

    Attributes:
        enabled: Whether to include tool chaos in fuzzing.
        probability: Probability weight for selecting tool chaos (0.0-1.0).
        faults: List of fault types to choose from.
            Options: "error", "timeout", "empty"
        targets: List of tool names to target. If None and fuzz_any=True,
            chaos will be generated without a specific tool name.
        fuzz_any: If True and targets is None, generate tool chaos without
            specifying a tool name.
        mutators: User-provided tool mutation functions to randomly select from.
        mutator_probability: Probability weight for selecting a tool mutator.
        error_messages: Custom error messages for error faults.
        timeout_range: Range for timeout_seconds parameter (seconds).

    Example:
        tool = ToolFuzzConfig(
            targets=["lookup_order", "process_refund"],
            mutators=[corrupt_order_status, corrupt_refund_amount],
            mutator_probability=0.3,
        )
    """

    enabled: bool = True
    probability: float = 0.3
    faults: list[str] = field(default_factory=lambda: ["error", "timeout", "empty"])
    targets: list[str] | None = None
    fuzz_any: bool = False
    mutators: list[ToolMutator | ToolMutatorWithCtx] = field(default_factory=list)
    mutator_probability: float = 0.2
    error_messages: list[str] = field(
        default_factory=lambda: [
            "Service unavailable",
            "Internal error",
            "Connection refused",
            "Request timeout",
            "Rate limit exceeded",
        ]
    )
    timeout_range: tuple[float, float] = (5.0, 60.0)

    @classmethod
    def disabled(cls) -> "ToolFuzzConfig":
        """Create a disabled tool fuzz config."""
        return cls(enabled=False)

    @classmethod
    def for_tools(cls, tools: list[str], **kwargs) -> "ToolFuzzConfig":
        """Create a tool fuzz config targeting specific tools."""
        return cls(targets=tools, **kwargs)

    @classmethod
    def heavy(cls, tools: list[str] | None = None) -> "ToolFuzzConfig":
        """Create a heavy tool fuzz config for stress testing."""
        return cls(
            probability=0.5,
            targets=tools,
            fuzz_any=tools is None,
        )


@dataclass
class ContextFuzzConfig:
    """Configuration for context mutation chaos fuzzing.

    Attributes:
        enabled: Whether to include context mutation chaos in fuzzing.
        probability: Probability weight for selecting context mutation.
        mutators: User-provided context mutation functions to randomly select from.

    Example:
        context = ContextFuzzConfig(
            enabled=True,
            probability=0.2,
            mutators=[inject_system_message, truncate_history],
        )
    """

    enabled: bool = False
    probability: float = 0.1
    mutators: list[ContextMutator | ContextMutatorWithCtx] = field(default_factory=list)

    @classmethod
    def disabled(cls) -> "ContextFuzzConfig":
        """Create a disabled context fuzz config."""
        return cls(enabled=False)

    @classmethod
    def with_mutators(
        cls, mutators: list[ContextMutator | ContextMutatorWithCtx], probability: float = 0.2
    ) -> "ContextFuzzConfig":
        """Create a context fuzz config with mutators."""
        return cls(enabled=True, probability=probability, mutators=mutators)


# =============================================================================
# Main ChaosSpace Configuration
# =============================================================================


@dataclass
class ChaosSpace:
    """Configuration space for random chaos generation.

    Organizes chaos fuzzing configuration into logical groups:
    - llm: LLM-related faults (rate limits, timeouts, errors)
    - stream: Stream-related faults (cuts, hangs, delays)
    - tool: Tool-related faults (errors, timeouts, empty responses)
    - context: Context mutations

    Attributes:
        llm: Configuration for LLM chaos fuzzing.
        stream: Configuration for stream chaos fuzzing.
        tool: Configuration for tool chaos fuzzing.
        context: Configuration for context mutation fuzzing.
        min_per_scenario: Minimum number of chaos to inject per scenario.
        max_per_scenario: Maximum number of chaos to inject per scenario.
        after_calls_range: Range for after_calls trigger (min, max).

    Example:
        # Default configuration
        space = ChaosSpace()

        # LLM-focused testing
        space = ChaosSpace.llm_focused()

        # Custom configuration with nested configs
        space = ChaosSpace(
            llm=LLMFuzzConfig(probability=0.5),
            tool=ToolFuzzConfig(
                targets=["lookup_order"],
                mutators=[my_mutator],
            ),
            min_per_scenario=2,
            max_per_scenario=4,
        )
    """

    llm: LLMFuzzConfig = field(default_factory=LLMFuzzConfig)
    stream: StreamFuzzConfig = field(default_factory=StreamFuzzConfig)
    tool: ToolFuzzConfig = field(default_factory=ToolFuzzConfig)
    context: ContextFuzzConfig = field(default_factory=ContextFuzzConfig)
    min_per_scenario: int = 1
    max_per_scenario: int = 3
    after_calls_range: tuple[int, int] = (1, 5)

    @classmethod
    def default(cls) -> "ChaosSpace":
        """Default configuration with balanced chaos distribution."""
        return cls()

    @classmethod
    def llm_focused(cls) -> "ChaosSpace":
        """Heavy LLM faults, minimal tool/stream faults."""
        return cls(
            llm=LLMFuzzConfig.heavy(),
            stream=StreamFuzzConfig(probability=0.2),
            tool=ToolFuzzConfig.disabled(),
        )

    @classmethod
    def stream_focused(cls) -> "ChaosSpace":
        """Heavy stream faults for testing streaming resilience."""
        return cls(
            llm=LLMFuzzConfig(probability=0.1),
            stream=StreamFuzzConfig.heavy(),
            tool=ToolFuzzConfig(probability=0.1),
        )

    @classmethod
    def tool_focused(cls, tools: list[str]) -> "ChaosSpace":
        """Heavy tool faults for testing tool error handling."""
        return cls(
            llm=LLMFuzzConfig(probability=0.1),
            stream=StreamFuzzConfig(probability=0.1),
            tool=ToolFuzzConfig(probability=0.6, targets=tools),
        )

    @classmethod
    def stress(cls, tools: list[str] | None = None) -> "ChaosSpace":
        """High chaos density for stress testing."""
        return cls(
            llm=LLMFuzzConfig.heavy(),
            stream=StreamFuzzConfig.heavy(),
            tool=ToolFuzzConfig.heavy(tools),
            min_per_scenario=2,
            max_per_scenario=5,
        )


# =============================================================================
# Fuzz Generation Functions
# =============================================================================


def fuzz_chaos(
    scenario: "Scenario",
    n: int = 10,
    seed: int | None = None,
    space: ChaosSpace | None = None,
    tag: str = "fuzz",
) -> list["Scenario"]:
    """Generate N scenario variations with random chaos configurations.

    Takes a base scenario and generates N variations, each with different
    randomly-selected chaos injected. The generated scenarios share the
    same turns, agent, and assertions as the base scenario.

    Args:
        scenario: Base scenario to fuzz. Must have at least one turn.
        n: Number of variations to generate.
        seed: Random seed for reproducibility. If None, uses random seed.
        space: Chaos configuration space. Defaults to ChaosSpace.default().
        tag: Tag to add to generated scenarios for filtering.

    Returns:
        List of N scenarios with random chaos injected.
        Each scenario has:
        - name: "{original_name}--fuzz-{index}"
        - tags: [tag] appended to existing tags
        - description: Contains seed and chaos info

    Example:
        baseline = Scenario(
            name="order-check",
            agent=my_agent,
            turns=[Turn("Check order #123")],
            assertions=[AllTurnsComplete()],
        )

        # Generate 20 fuzzed variations
        fuzzed = fuzz_chaos(baseline, n=20, seed=42)

        def get_scenarios():
            return [baseline] + fuzzed

        # Or with custom space
        fuzzed = fuzz_chaos(
            baseline,
            n=20,
            seed=42,
            space=ChaosSpace(
                tool=ToolFuzzConfig(
                    targets=["lookup_order", "process_refund"],
                    mutators=[corrupt_order_data],
                ),
                min_per_scenario=1,
                max_per_scenario=2,
            ),
        )
    """
    from agent_chaos.scenario.model import Scenario as ScenarioClass

    space = space or ChaosSpace.default()

    # Initialize RNG
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)

    # Validate scenario
    num_turns = len(scenario.turns)
    if num_turns == 0:
        raise ValueError("Scenario must have at least one turn to fuzz")

    scenarios: list[Scenario] = []

    for i in range(n):
        # Generate random chaos for this variation
        chaos_list = _generate_random_chaos(rng, space, num_turns)

        # Create a description of generated chaos for metadata
        # Build each chaos to get proper string representation
        chaos_descriptions = [str(c.build()) for c in chaos_list]

        # Build readable multiline description
        chaos_bullets = "\n".join(f"  • {c}" for c in chaos_descriptions)
        description = (
            f"Fuzzed variation {i} of {scenario.name}\n"
            f"Seed: {seed}\n"
            f"Chaos:\n{chaos_bullets}"
        )

        # Create the fuzzed scenario
        fuzzed = ScenarioClass(
            name=f"{scenario.name}--fuzz-{i}",
            description=description,
            agent=scenario.agent,
            turns=deepcopy(scenario.turns),  # Deep copy to avoid mutation
            chaos=list(scenario.chaos) + chaos_list,  # Base chaos + fuzzed chaos
            providers=scenario.providers,
            assertions=list(scenario.assertions),
            tags=list(scenario.tags) + [tag],
        )

        scenarios.append(fuzzed)

    return scenarios


def _generate_random_chaos(
    rng: random.Random,
    space: ChaosSpace,
    num_turns: int,
) -> list[ChaosBuilder]:
    """Generate a random chaos configuration.

    Args:
        rng: Random number generator.
        space: Chaos configuration space.
        num_turns: Number of turns in the scenario (for valid turn ranges).

    Returns:
        List of ChaosBuilder instances representing the random chaos.
    """
    chaos_list: list[ChaosBuilder] = []
    target_count = rng.randint(space.min_per_scenario, space.max_per_scenario)

    # Track what we've generated to avoid duplicates
    generated: set[str] = set()

    attempts = 0
    max_attempts = target_count * 5  # Prevent infinite loops

    while len(chaos_list) < target_count and attempts < max_attempts:
        attempts += 1

        chaos = _maybe_generate_one(rng, space, num_turns, generated)
        if chaos is not None:
            chaos_list.append(chaos)
            generated.add(_chaos_key(chaos))

    return chaos_list


def _chaos_key(chaos: ChaosBuilder) -> str:
    """Generate a key for deduplication."""
    # Use string representation as key
    return str(chaos.build())


def _maybe_generate_one(
    rng: random.Random,
    space: ChaosSpace,
    num_turns: int,
    generated: set[str],
) -> ChaosBuilder | None:
    """Try to generate one random chaos.

    Args:
        rng: Random number generator.
        space: Chaos configuration space.
        num_turns: Number of turns.
        generated: Set of already generated chaos keys.

    Returns:
        A ChaosBuilder instance, or None if nothing was generated.
    """
    # Build list of possible categories with their probabilities
    categories: list[tuple[str, float]] = []

    if space.llm.enabled and space.llm.faults:
        categories.append(("llm", space.llm.probability))
    if space.stream.enabled and space.stream.faults:
        categories.append(("stream", space.stream.probability))
    if space.tool.enabled and (space.tool.targets or space.tool.fuzz_any):
        categories.append(("tool", space.tool.probability))
    if space.tool.enabled and space.tool.mutators:
        categories.append(("tool_mutator", space.tool.mutator_probability))
    if space.context.enabled and space.context.mutators:
        categories.append(("context_mutator", space.context.probability))

    if not categories:
        return None

    # Weighted random selection
    total = sum(p for _, p in categories)
    if total <= 0:
        return None

    r = rng.random() * total
    selected: str | None = None

    for cat, prob in categories:
        r -= prob
        if r <= 0:
            selected = cat
            break

    if selected is None:
        selected = categories[-1][0]

    # Generate random turn and call triggers
    turn = rng.randint(1, num_turns)
    call = rng.randint(*space.after_calls_range)

    # Generate chaos based on selected category
    if selected == "llm":
        return _generate_llm_chaos(rng, space.llm, turn, call)

    elif selected == "stream":
        return _generate_stream_chaos(rng, space.stream, turn)

    elif selected == "tool":
        return _generate_tool_chaos(rng, space.tool, turn)

    elif selected == "tool_mutator":
        return _generate_tool_mutator_chaos(rng, space.tool, turn)

    elif selected == "context_mutator":
        return _generate_context_mutator_chaos(rng, space.context, turn)

    return None


def _generate_llm_chaos(
    rng: random.Random,
    config: LLMFuzzConfig,
    turn: int,
    call: int,
) -> ChaosBuilder:
    """Generate random LLM chaos."""
    fault_type = rng.choice(config.faults)
    chaos_class = _LLM_FAULTS[fault_type]

    # Create builder with appropriate defaults
    kwargs: dict[str, Any] = {}

    if fault_type == "rate_limit":
        kwargs["retry_after"] = rng.uniform(*config.retry_after_range)
    elif fault_type == "timeout":
        kwargs["delay"] = rng.uniform(*config.timeout_range)

    builder = ChaosBuilder(chaos_class, **kwargs)
    return builder.on_turn(turn).after_calls(call)


def _generate_stream_chaos(
    rng: random.Random,
    config: StreamFuzzConfig,
    turn: int,
) -> ChaosBuilder:
    """Generate random stream chaos."""
    fault_type = rng.choice(config.faults)
    chaos_class = _STREAM_FAULTS[fault_type]

    kwargs: dict[str, Any] = {}

    if fault_type in ("stream_cut", "stream_hang"):
        kwargs["after_chunks"] = rng.randint(*config.cut_after_chunks)
    elif fault_type == "slow_ttft":
        kwargs["delay"] = rng.uniform(*config.ttft_delay_range)
    elif fault_type == "slow_chunks":
        kwargs["delay"] = rng.uniform(*config.chunk_delay_range)

    builder = ChaosBuilder(chaos_class, **kwargs)
    return builder.on_turn(turn)


def _generate_tool_chaos(
    rng: random.Random,
    config: ToolFuzzConfig,
    turn: int,
) -> ChaosBuilder:
    """Generate random tool chaos."""
    fault_type = rng.choice(config.faults)
    chaos_class = _TOOL_FAULTS[fault_type]

    kwargs: dict[str, Any] = {}

    if fault_type == "error":
        kwargs["message"] = rng.choice(config.error_messages)
    elif fault_type == "timeout":
        kwargs["timeout_seconds"] = rng.uniform(*config.timeout_range)

    builder = ChaosBuilder(chaos_class, **kwargs)
    builder = builder.on_turn(turn)

    # Assign to specific tool if available
    if config.targets:
        tool = rng.choice(config.targets)
        builder = builder.for_tool(tool)

    return builder


def _generate_tool_mutator_chaos(
    rng: random.Random,
    config: ToolFuzzConfig,
    turn: int,
) -> ChaosBuilder:
    """Generate random tool mutator chaos using user-provided functions."""
    mutator = rng.choice(config.mutators)

    builder = ChaosBuilder(ToolMutateChaos, mutator=mutator)
    builder = builder.on_turn(turn)

    # Assign to specific tool if available
    if config.targets:
        tool = rng.choice(config.targets)
        builder = builder.for_tool(tool)

    return builder


def _generate_context_mutator_chaos(
    rng: random.Random,
    config: ContextFuzzConfig,
    turn: int,
) -> ChaosBuilder:
    """Generate random context mutator chaos using user-provided functions."""
    mutator = rng.choice(config.mutators)

    builder = ChaosBuilder(ContextMutateChaos, mutator=mutator)
    return builder.on_turn(turn)


# Convenience function for inline usage
def fuzz(
    n: int = 10,
    seed: int | None = None,
    space: ChaosSpace | None = None,
) -> Callable[["Scenario"], list["Scenario"]]:
    """Create a fuzzer function for use in scenario definitions.

    This is a convenience wrapper around fuzz_chaos() for cleaner syntax
    when defining scenarios inline.

    Args:
        n: Number of variations to generate.
        seed: Random seed for reproducibility.
        space: Chaos configuration space.

    Returns:
        A function that takes a Scenario and returns fuzzed variations.

    Example:
        fuzzer = fuzz(n=20, seed=42)
        fuzzed_scenarios = fuzzer(my_scenario)

        # Or inline
        def get_scenarios():
            return [baseline] + fuzz(n=20, seed=42)(baseline)
    """

    def fuzzer(scenario: "Scenario") -> list["Scenario"]:
        return fuzz_chaos(scenario, n=n, seed=seed, space=space)

    return fuzzer
