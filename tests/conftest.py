"""Pytest configuration shared by the whole test suite.

Adds two directories to ``sys.path`` so the tests run without
requiring an editable ``pip install``:

  * The repository root, so ``from tests.helpers import ...`` works.
  * ``src/``, so ``import maxdiff`` resolves to the live source.

CI installs the dev requirements but does not run ``pip install -e .``;
running tests from a fresh checkout should work the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

for path in (_REPO_ROOT, _SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
