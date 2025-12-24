"""Scenario loader for CLI.

Supported refs:
- `path/to/file.py` (expects `scenario` variable or `get_scenario()` function)
- `package.module:attr` (attr is Scenario or callable returning Scenario)
- directory path: loads all `*.py` files as scenario modules
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_chaos.scenario.model import Scenario


def _load_module_from_file(path: Path) -> ModuleType:
    """Load a scenario module from a file path via importlib.

    Important: we register the module in `sys.modules` *before* executing it.
    Some decorators (notably `dataclasses.dataclass`) require this invariant.
    """
    # Use a unique name to avoid collisions when loading files with the same stem.
    suffix = f"{abs(hash(str(path))) & 0xFFFFFFFF:x}"
    module_name = f"agent_chaos_scenario_{path.stem}_{suffix}"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for scenario: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _coerce_scenario(obj: Any) -> Scenario:
    if isinstance(obj, Scenario):
        return obj
    if callable(obj):
        v = obj()
        if isinstance(v, Scenario):
            return v
    raise TypeError(
        "Scenario reference must resolve to `Scenario` or a callable returning `Scenario`"
    )


def _coerce_scenarios(obj: Any) -> list[Scenario]:
    """Coerce an object into a list of Scenario."""
    if isinstance(obj, Scenario):
        return [obj]
    if isinstance(obj, list) and all(isinstance(s, Scenario) for s in obj):
        return obj
    if callable(obj):
        v = obj()
        if isinstance(v, Scenario):
            return [v]
        if isinstance(v, list) and all(isinstance(s, Scenario) for s in v):
            return v
    raise TypeError(
        "Scenario reference must resolve to `Scenario`, `list[Scenario]`, or a callable returning one of those"
    )


def load_target(ref: str) -> list[Scenario]:
    """Load one or more scenarios from a ref.

    Supports:
    - file.py: scenario/get_scenario/scenarios/get_scenarios
    - module:attr where attr is Scenario / list[Scenario] or callable returning them
    """
    # module:attr form
    if ":" in ref and not ref.strip().endswith(".py"):
        mod_name, attr = ref.split(":", 1)
        module = importlib.import_module(mod_name)
        return _coerce_scenarios(getattr(module, attr))

    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(ref)
    if path.is_dir():
        raise IsADirectoryError(ref)

    module = _load_module_from_file(path.resolve())
    if hasattr(module, "scenarios"):
        return _coerce_scenarios(getattr(module, "scenarios"))
    if hasattr(module, "get_scenarios"):
        return _coerce_scenarios(getattr(module, "get_scenarios"))
    if hasattr(module, "scenario"):
        return _coerce_scenarios(getattr(module, "scenario"))
    if hasattr(module, "get_scenario"):
        return _coerce_scenarios(getattr(module, "get_scenario"))

    raise AttributeError(
        f"{ref} must define `scenario`, `get_scenario()`, `scenarios`, or `get_scenarios()`"
    )


def load_scenario(ref: str) -> Scenario:
    # module:attr form
    if ":" in ref and not ref.strip().endswith(".py"):
        mod_name, attr = ref.split(":", 1)
        module = importlib.import_module(mod_name)
        return _coerce_scenario(getattr(module, attr))

    # file path form
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(ref)
    if path.is_dir():
        raise IsADirectoryError(
            f"{ref} is a directory; use load_scenarios_from_dir() or the CLI `run-suite` command"
        )
    module = _load_module_from_file(path.resolve())

    if hasattr(module, "scenario"):
        return _coerce_scenario(getattr(module, "scenario"))
    if hasattr(module, "get_scenario"):
        return _coerce_scenario(getattr(module, "get_scenario"))

    raise AttributeError(
        f"{ref} must define `scenario: Scenario` or `def get_scenario() -> Scenario`"
    )


def load_scenarios_from_dir(
    dir_path: str | Path,
    *,
    glob: str = "*.py",
    recursive: bool = False,
) -> list[Scenario]:
    """Discover and load multiple scenarios from a directory.

    Each matching python file must define:
    - `scenario: Scenario`, or
    - `def get_scenario() -> Scenario`
    """
    base = Path(dir_path)
    if not base.exists():
        raise FileNotFoundError(str(dir_path))
    if not base.is_dir():
        raise NotADirectoryError(str(dir_path))

    pattern = f"**/{glob}" if recursive else glob
    scenarios: list[Scenario] = []

    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        if path.name.startswith("_"):
            continue
        if path.name == "__init__.py":
            continue
        scenarios.extend(load_target(str(path)))

    if not scenarios:
        raise FileNotFoundError(
            f"No scenarios found in {base} (glob={glob}, recursive={recursive})"
        )

    return scenarios
