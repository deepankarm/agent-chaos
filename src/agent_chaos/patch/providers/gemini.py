"""Gemini provider patcher (placeholder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_chaos.patch.base import BaseProviderPatcher

if TYPE_CHECKING:
    from agent_chaos.core.injector import ChaosInjector
    from agent_chaos.core.metrics import MetricsStore


class GeminiPatcher(BaseProviderPatcher):
    """Patches Google Gemini SDK methods to inject chaos."""

    provider_name = "gemini"

    def patch(self, injector: "ChaosInjector", metrics: "MetricsStore") -> None:
        """Apply patches to Gemini SDK."""
        raise NotImplementedError("Gemini provider not yet implemented")
