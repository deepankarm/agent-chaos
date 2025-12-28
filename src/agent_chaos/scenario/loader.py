"""Scenario loader for CLI.

Supported refs:
- `path/to/file.py` (expects `scenario` variable or `get_scenario()` function)
- `package.module:attr` (attr is Scenario or callable returning Scenario)
- directory path: loads the package and calls get_scenarios() or scenarios
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
    """Load a scenario module from a file path via importlib."""
    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

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


def _load_package(dir_path: Path) -> ModuleType | None:
    """Try to load a directory as a Python package.

    Returns the module if it has __init__.py with get_scenarios/scenarios,
    otherwise returns None.
    """
    init_file = dir_path / "__init__.py"
    if not init_file.exists():
        return None

    # Add parent to sys.path so we can import the package
    parent = str(dir_path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Also add cwd for relative imports within the package
    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    package_name = dir_path.name

    try:
        # Try to import as a package
        module = importlib.import_module(package_name)

        # Check if it has scenario-related exports
        if hasattr(module, "get_scenarios") or hasattr(module, "scenarios"):
            return module
        return None
    except ImportError:
        return None


def load_scenarios_from_dir(
    dir_path: str | Path,
    *,
    glob: str = "*.py",
    recursive: bool = False,
) -> list[Scenario]:
    """Discover and load multiple scenarios from a directory.

    If the directory is a Python package (has __init__.py with get_scenarios),
    it imports the package and calls get_scenarios().

    Otherwise, each matching python file must define:
    - `scenario: Scenario`, or
    - `def get_scenario() -> Scenario`
    """
    base = Path(dir_path).resolve()
    if not base.exists():
        raise FileNotFoundError(str(dir_path))
    if not base.is_dir():
        raise NotADirectoryError(str(dir_path))

    # First, try to load as a package
    package_module = _load_package(base)
    if package_module is not None:
        if hasattr(package_module, "get_scenarios"):
            return _coerce_scenarios(getattr(package_module, "get_scenarios"))
        if hasattr(package_module, "scenarios"):
            return _coerce_scenarios(getattr(package_module, "scenarios"))

    # Fall back to loading individual files
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


def load_scenarios(
    targets: list[str], glob: str = "*.py", recursive: bool = False
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for t in targets:
        p = Path(t)
        if p.exists() and p.is_dir():
            scenarios.extend(load_scenarios_from_dir(p, glob=glob, recursive=recursive))
        else:
            scenarios.extend(load_target(t))
    return scenarios
