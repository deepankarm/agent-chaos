"""OpenAI provider patcher (placeholder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_chaos.patch.base import BaseProviderPatcher

if TYPE_CHECKING:
    from agent_chaos.core.injector import ChaosInjector
    from agent_chaos.core.metrics import MetricsStore


class OpenAIPatcher(BaseProviderPatcher):
    """Patches OpenAI SDK methods to inject chaos."""

    provider_name = "openai"

    def patch(self, injector: "ChaosInjector", metrics: "MetricsStore") -> None:
        """Apply patches to OpenAI SDK."""
        raise NotImplementedError("OpenAI provider not yet implemented")
