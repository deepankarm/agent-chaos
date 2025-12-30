"""Core patcher for monkeypatching SDK methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_chaos.patch.base import BaseProviderPatcher
from agent_chaos.patch.discovery import get_available_providers, load_providers

if TYPE_CHECKING:
    from agent_chaos.core.injector import ChaosInjector
    from agent_chaos.core.metrics import MetricsStore


class ChaosPatcher:
    """Patches SDK methods to inject chaos.

    This is the main orchestrator that delegates to provider-specific patchers.
    Providers are lazily loaded based on what's installed.
    """

    def __init__(self, injector: "ChaosInjector", metrics: "MetricsStore"):
        self.injector = injector
        self.metrics = metrics
        self._providers: list[BaseProviderPatcher] = []
        self._patched = False

    def patch_all(self):
        """Patch all installed providers."""
        if self._patched:
            return
        for provider in self._providers:
            provider.patch(self.injector, self.metrics)
        self._patched = True

    def patch_providers(self, provider_names: list[str]):
        """Patch specific providers by name.

        Args:
            provider_names: List of provider names to patch (e.g., ["anthropic", "openai"])
        """
        if self._patched:
            return
        self._providers = load_providers(provider_names)
        for provider in self._providers:
            provider.patch(self.injector, self.metrics)
        self._patched = True

    def unpatch_all(self):
        """Restore original methods for all patched providers."""
        for provider in self._providers:
            provider.unpatch()
        self._providers.clear()
        self._patched = False

    @staticmethod
    def available_providers() -> list[str]:
        """Return names of providers that are installed.

        Returns:
            List of provider names (e.g., ["anthropic", "openai"])
        """
        return get_available_providers()
