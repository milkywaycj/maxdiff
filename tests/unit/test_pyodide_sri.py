"""Verify the pinned Pyodide bootstrap SHA-384 hash.

The browser analysis tool fetches ``pyodide.js`` from the jsDelivr CDN
and refuses to evaluate it unless the bytes match a pinned SHA-384
hash. The hash is stored as a constant in ``docs/analysis/index.html``.

This test reads the pinned hash from the HTML, downloads the same
versioned pyodide.js from the CDN, and asserts the live file's hash
still matches. Two failure modes are useful to surface:

* If the CDN response is tampered with or replaced, the hash mismatches
  and CI fails - exactly the situation the SRI check is designed to
  catch.

* If a developer bumps PYODIDE_VERSION without also updating the hash,
  this test fails with a clear diff, preventing a silent regression
  where users see "Pyodide integrity check failed" in their browser.

The test is marked ``slow`` so it can be excluded from quick local
runs (``pytest -m 'not slow'``).
"""

from __future__ import annotations

import base64
import hashlib
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_HTML = _REPO_ROOT / "docs" / "analysis" / "index.html"

_VERSION_RE = re.compile(r"const PYODIDE_VERSION\s*=\s*['\"]([^'\"]+)['\"]")
_HASH_RE = re.compile(r"const PYODIDE_BOOTSTRAP_SHA384\s*=\s*['\"]([^'\"]+)['\"]")


def _read_pinned_constants() -> tuple[str, str]:
    """Return (version, sha384_b64) read from the analysis HTML page."""
    text = _ANALYSIS_HTML.read_text(encoding="utf-8")
    version_match = _VERSION_RE.search(text)
    hash_match = _HASH_RE.search(text)
    assert version_match, "PYODIDE_VERSION constant missing from docs/analysis/index.html"
    assert hash_match, "PYODIDE_BOOTSTRAP_SHA384 constant missing from docs/analysis/index.html"
    return version_match.group(1), hash_match.group(1)


def test_pinned_constants_are_well_formed() -> None:
    """Cheap check that runs without network access."""
    version, sha = _read_pinned_constants()
    # Version should look like vN.N[.N]
    assert re.match(r"^v\d+\.\d+(\.\d+)?$", version), f"Unexpected version format: {version}"
    # SHA-384 base64: 384 bits = 48 bytes -> 64 base64 chars without padding,
    # or 64 with up to 2 trailing '='. Account for jsDelivr-style hashes.
    decoded = base64.b64decode(sha + "==")
    assert len(decoded) == 48, f"Pinned hash decodes to {len(decoded)} bytes; expected 48 (SHA-384)"


@pytest.mark.slow
def test_pinned_hash_matches_cdn_content() -> None:
    """Hit the CDN and recompute SHA-384. Slow because of the network
    round trip; excluded from `pytest -m 'not slow'` runs."""
    version, pinned_sha = _read_pinned_constants()
    url = f"https://cdn.jsdelivr.net/pyodide/{version}/full/pyodide.js"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
    except urllib.error.URLError as exc:
        pytest.skip(f"CDN unreachable: {exc}")

    digest = hashlib.sha384(content).digest()
    computed = base64.b64encode(digest).decode("ascii")

    assert computed == pinned_sha, (
        f"\nPinned Pyodide hash no longer matches CDN content.\n"
        f"  URL:        {url}\n"
        f"  pinned:     {pinned_sha}\n"
        f"  CDN now is: {computed}\n"
        f"This is either an intentional Pyodide bump (update the constant in "
        f"docs/analysis/index.html and this test will pass) or a real CDN tampering "
        f"event (the SRI check has prevented a privacy regression - investigate)."
    )
