"""Lazy provider discovery - only load what's installed."""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_chaos.patch.base import BaseProviderPatcher

# Registry: provider_name -> (module_path, class_name, required_package)
# The required_package is the root package to check for installation
PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "anthropic": (
        "agent_chaos.patch.providers.anthropic",
        "AnthropicPatcher",
        "anthropic",
    ),
    "openai": (
        "agent_chaos.patch.providers.openai",
        "OpenAIPatcher",
        "openai",
    ),
    "gemini": (
        "agent_chaos.patch.providers.gemini",
        "GeminiPatcher",
        "google.generativeai",
    ),
}


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed without importing it.

    Args:
        package_name: Package name (e.g., "anthropic" or "google.generativeai")

    Returns:
        True if the package is installed.
    """
    # Handle nested package names like "google.generativeai"
    root_package = package_name.split(".")[0]
    return importlib.util.find_spec(root_package) is not None


def get_available_providers() -> list[str]:
    """Return list of provider names that are installed.

    Returns:
        List of provider names (e.g., ["anthropic", "openai"])
    """
    return [
        name
        for name, (_, _, pkg) in PROVIDER_REGISTRY.items()
        if is_package_installed(pkg)
    ]


def load_provider(name: str) -> "BaseProviderPatcher":
    """Lazily load and instantiate a provider patcher.

    Args:
        name: Provider name (e.g., "anthropic")

    Returns:
        Instantiated provider patcher.

    Raises:
        ValueError: If provider name is unknown.
        ImportError: If required package is not installed.
    """
    if name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")

    module_path, class_name, required_pkg = PROVIDER_REGISTRY[name]

    if not is_package_installed(required_pkg):
        raise ImportError(
            f"Provider '{name}' requires '{required_pkg}'. "
            f"Install with: uv add agent-chaos[{name}]"
        )

    module = importlib.import_module(module_path)
    patcher_class = getattr(module, class_name)
    return patcher_class()


def load_providers(names: list[str]) -> list["BaseProviderPatcher"]:
    """Load specific providers by name.

    Only loads providers that are both requested AND installed.
    Silently skips providers that are not installed.

    Args:
        names: List of provider names to load.

    Returns:
        List of instantiated provider patchers.
    """
    providers = []
    for name in names:
        if name not in PROVIDER_REGISTRY:
            continue
        _, _, required_pkg = PROVIDER_REGISTRY[name]
        if not is_package_installed(required_pkg):
            continue
        try:
            providers.append(load_provider(name))
        except Exception:
            pass
    return providers


def load_all_available_providers() -> list["BaseProviderPatcher"]:
    """Load all providers that are currently installed.

    Returns:
        List of instantiated provider patchers.
    """
    return load_providers(list(PROVIDER_REGISTRY.keys()))
