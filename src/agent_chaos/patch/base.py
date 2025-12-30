"""Base class for provider patchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent_chaos.core.injector import ChaosInjector
    from agent_chaos.core.metrics import MetricsStore


class BaseProviderPatcher(ABC):
    """Abstract base class for provider-specific patchers.

    Each provider (anthropic, openai, gemini) implements this interface
    to handle its own SDK patching logic.
    """

    # Subclasses must set this
    provider_name: str = ""

    def __init__(self):
        self._original_methods: dict[str, Callable] = {}
        self._patched = False

    @abstractmethod
    def patch(self, injector: "ChaosInjector", metrics: "MetricsStore") -> None:
        """Apply patches to the provider's SDK.

        Args:
            injector: The chaos injector instance.
            metrics: The metrics store instance.
        """
        ...

    def unpatch(self) -> None:
        """Restore original methods."""
        for path, original in self._original_methods.items():
            self._set_method(path, original)
        self._original_methods.clear()
        self._patched = False

    def _save_original(self, path: str, method: Callable) -> None:
        """Save original method for restoration."""
        if path not in self._original_methods:
            self._original_methods[path] = method

    def _set_method(self, path: str, method: Callable) -> None:
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
