"""Verify the pinned Pyodide bootstrap SHA-384 hash against vendored bytes.

The browser analysis tool serves Pyodide same-origin from
``docs/vendor/pyodide-<version>/``. The bootstrap loader is still
SHA-384-checked inside the worker as defense-in-depth: if the vendored
``pyodide.js`` is ever altered in the repo without bumping the pinned
constant, this test fails before the change can ship.

The hash is read from ``docs/analysis/index.html``; the vendored file is
read from ``docs/vendor/pyodide-<version>/pyodide.js``. No network access
is required.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_HTML = _REPO_ROOT / "docs" / "analysis" / "index.html"
_VENDOR_ROOT = _REPO_ROOT / "docs" / "vendor"

_VERSION_RE = re.compile(r"const PYODIDE_VERSION\s*=\s*['\"]([^'\"]+)['\"]")
_HASH_RE = re.compile(r"const PYODIDE_BOOTSTRAP_SHA384\s*=\s*['\"]([^'\"]+)['\"]")


def _read_pinned_constants() -> tuple[str, str]:
    """Return ``(version, sha384_b64)`` read from the analysis HTML page."""
    text = _ANALYSIS_HTML.read_text(encoding="utf-8")
    version_match = _VERSION_RE.search(text)
    hash_match = _HASH_RE.search(text)
    assert version_match, "PYODIDE_VERSION constant missing from docs/analysis/index.html"
    assert hash_match, "PYODIDE_BOOTSTRAP_SHA384 constant missing from docs/analysis/index.html"
    return version_match.group(1), hash_match.group(1)


def test_pinned_constants_are_well_formed() -> None:
    """Cheap structural check on the pinned constants."""
    version, sha = _read_pinned_constants()
    assert re.match(r"^v\d+\.\d+(\.\d+)?$", version), f"Unexpected version format: {version}"
    decoded = base64.b64decode(sha + "==")
    assert len(decoded) == 48, f"Pinned hash decodes to {len(decoded)} bytes; expected 48 (SHA-384)"


def test_pinned_hash_matches_vendored_pyodide_js() -> None:
    """The vendored ``pyodide.js`` must hash to the pinned SHA-384.

    Running this is the offline equivalent of the in-browser integrity
    check: if someone alters ``docs/vendor/pyodide-<version>/pyodide.js``
    without updating ``PYODIDE_BOOTSTRAP_SHA384`` (or vice versa), the
    browser tool would refuse to start at runtime. This test catches
    that mismatch in CI.
    """
    version, pinned_sha = _read_pinned_constants()
    vendored_js = _VENDOR_ROOT / f"pyodide-{version.lstrip('v')}" / "pyodide.js"

    assert vendored_js.exists(), (
        f"Expected vendored Pyodide bootstrap at {vendored_js.relative_to(_REPO_ROOT)}. "
        f"Run `python scripts/vendor_pyodide.py` to populate docs/vendor/."
    )

    content = vendored_js.read_bytes()
    digest = hashlib.sha384(content).digest()
    computed = base64.b64encode(digest).decode("ascii")

    assert computed == pinned_sha, (
        f"\nPinned Pyodide hash does not match vendored bytes.\n"
        f"  Vendored file: {vendored_js.relative_to(_REPO_ROOT)}\n"
        f"  pinned:        {pinned_sha}\n"
        f"  computed:      {computed}\n"
        "Either the pinned constant in docs/analysis/index.html is stale "
        "(intentional Pyodide bump? rerun scripts/vendor_pyodide.py then "
        "update the constant) or the vendored file has been tampered with."
    )
