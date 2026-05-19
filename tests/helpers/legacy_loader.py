"""Compatibility shim that returns the maxdiff package for test code.

Originally this loader path-imported the monolithic
``src/MaxDiff_Data_Analyzer_v2-1.py`` because the analyzer had no
real package structure. Phase 3 extracted the pure analysis code
into the ``maxdiff`` package, and most tests now resolve their
symbols through ``maxdiff`` directly.

We keep ``load_analyzer()`` as a single entry point so the existing
suite can continue to call ``legacy.<name>`` uniformly. New tests
should prefer ``import maxdiff`` directly.

This module no longer reads the legacy file. The legacy file still
exists (it carries the desktop GUI) but importing it pulls in
customtkinter / tkinter, which is undesirable in headless test
runs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import maxdiff

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_analyzer() -> ModuleType:
    """Return the ``maxdiff`` package.

    Named ``load_analyzer`` for backward-compatibility with the
    pre-Phase-3 suite. The returned module exposes every public
    analysis symbol the tests reference.
    """
    return maxdiff


def legacy_source_path() -> Path:
    """Return the path to the desktop entry-point script (still
    customtkinter-flavored)."""
    return _REPO_ROOT / "src" / "MaxDiff_Data_Analyzer_v2-1.py"
