"""Pytest configuration shared by the whole test suite.

Currently this file only exists to ensure the repository root is on
``sys.path`` so that ``from tests.helpers import legacy_loader`` works
without requiring an editable install. As the suite grows it will pick
up reusable fixtures (synthetic data, golden-output snapshots, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Place the repository root at the front of sys.path. Pytest already does
# this when an __init__.py exists alongside tests, but being explicit
# avoids surprises when tests are invoked from inside a packaging build
# or with rootdir overrides.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
