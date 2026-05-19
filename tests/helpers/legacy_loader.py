"""Load the legacy analyzer module by file path.

The analyzer currently lives at ``src/MaxDiff_Data_Analyzer_v2-1.py``.
The hyphen and version suffix in the filename make it non-importable
with normal ``import`` syntax, so we use :mod:`importlib` to load it
from path.

This indirection is temporary. Phase 3 of the refactor extracts the
analysis code into an installable ``maxdiff`` package, at which point
this loader becomes obsolete and the tests should import from
``maxdiff.*`` directly.

The loader caches the loaded module so repeated imports across many
tests do not re-parse the 2 400-line source file every time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

# Path resolution: this file is at tests/helpers/legacy_loader.py, so
# the repository root is three parents up.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_LEGACY_PATH: Final[Path] = _REPO_ROOT / "src" / "MaxDiff_Data_Analyzer_v2-1.py"
_MODULE_NAME: Final[str] = "maxdiff_legacy"

# Process-wide cache. ``importlib.util.spec_from_file_location`` is
# deterministic, so caching here is safe across tests.
_cached: ModuleType | None = None


def load_analyzer() -> ModuleType:
    """Return the legacy analyzer module, loading it on first call.

    Raises
    ------
    FileNotFoundError
        If the legacy source file does not exist at the expected path.
        This usually means the repository layout has changed and this
        loader needs to be updated.
    ImportError
        If the file is present but fails to load (syntax error, missing
        runtime dependency, etc.). The original exception is chained.
    """
    global _cached
    if _cached is not None:
        return _cached

    if not _LEGACY_PATH.is_file():
        raise FileNotFoundError(
            f"Legacy analyzer not found at {_LEGACY_PATH}. "
            "If the repository layout changed, update tests/helpers/legacy_loader.py."
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {_LEGACY_PATH}")

    module = importlib.util.module_from_spec(spec)
    # Register before exec so any internal ``import maxdiff_legacy`` works
    # (defensive; the legacy module does not currently import itself).
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - exercised only on broken trees
        # Make sure a half-loaded module is not left in sys.modules.
        sys.modules.pop(_MODULE_NAME, None)
        raise ImportError(f"Failed to load legacy analyzer: {exc}") from exc

    _cached = module
    return module


def legacy_source_path() -> Path:
    """Return the absolute path to the legacy source file."""
    return _LEGACY_PATH
