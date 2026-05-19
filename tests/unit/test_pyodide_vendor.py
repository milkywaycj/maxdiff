"""Invariants for the vendored Pyodide runtime under ``docs/vendor/``.

The browser analysis tool is fully same-origin: it loads Pyodide and
every wheel it needs from ``docs/vendor/pyodide-<version>/`` rather than
from a CDN. This module pins the invariants of that arrangement so a
regression (an accidental CDN URL slipping back into the HTML, a
vendored file being deleted, a manifest hash drifting from on-disk
bytes) fails CI immediately.

These tests do NOT hit the network. The slow CDN-round-trip check was
the previous defense and is no longer needed: the vendored files are in
the repository, so we can verify directly against them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_HTML = _REPO_ROOT / "docs" / "analysis" / "index.html"
_VENDOR_ROOT = _REPO_ROOT / "docs" / "vendor"

_VERSION_RE = re.compile(r"const PYODIDE_VERSION\s*=\s*['\"]([^'\"]+)['\"]")


def _read_version() -> str:
    m = _VERSION_RE.search(_ANALYSIS_HTML.read_text(encoding="utf-8"))
    assert m, "PYODIDE_VERSION missing from docs/analysis/index.html"
    return m.group(1)


@pytest.fixture(scope="module")
def vendor_dir() -> Path:
    version = _read_version()
    path = _VENDOR_ROOT / f"pyodide-{version.lstrip('v')}"
    if not path.is_dir():
        pytest.skip(
            f"Vendored Pyodide directory missing at {path.relative_to(_REPO_ROOT)}. "
            "Run `python scripts/vendor_pyodide.py` to populate it."
        )
    return path


@pytest.fixture(scope="module")
def manifest(vendor_dir: Path) -> dict:
    return json.loads((vendor_dir / "MANIFEST.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# HTML invariants


def test_analysis_html_has_no_jsdelivr_references() -> None:
    """Vendoring is only effective if the HTML never reaches out to a CDN.

    A `cdn.jsdelivr.net` or similar URL hardcoded anywhere in the
    shipping HTML would silently undo the same-origin guarantee, so we
    forbid any such reference outright. The pinned-CDN string only
    survives on a code path that's no longer reachable, but the test
    catches even that case.
    """
    text = _ANALYSIS_HTML.read_text(encoding="utf-8")
    for needle in ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"):
        assert needle not in text, (
            f"Found {needle!r} in docs/analysis/index.html. The browser "
            "tool is supposed to be fully same-origin via "
            "docs/vendor/pyodide-<version>/. Update the page (or "
            "explicitly justify the exception in this test) before "
            "shipping."
        )


def test_analysis_html_points_pyodide_at_local_vendor_dir() -> None:
    """The PYODIDE_INDEX_URL expression should reference the vendor path."""
    text = _ANALYSIS_HTML.read_text(encoding="utf-8")
    # We don't try to evaluate the JS; we look for the literal
    # substring that builds the URL.
    assert "../vendor/pyodide-" in text, (
        "Expected docs/analysis/index.html to construct PYODIDE_INDEX_URL "
        "from `../vendor/pyodide-<version>/`. Did the vendoring get "
        "unwound?"
    )


# ---------------------------------------------------------------------------
# MANIFEST invariants


def test_manifest_records_pyodide_version(manifest: dict) -> None:
    version = _read_version()
    assert manifest["pyodide_version"] == version, (
        f"MANIFEST.json says version={manifest['pyodide_version']!r} but "
        f"docs/analysis/index.html pins PYODIDE_VERSION={version!r}. "
        "Re-run scripts/vendor_pyodide.py."
    )


def test_manifest_includes_pyodide_bootstrap_files(manifest: dict) -> None:
    required = {
        "pyodide.js",
        "pyodide.asm.js",
        "pyodide.asm.wasm",
        "python_stdlib.zip",
        "pyodide-lock.json",
    }
    missing = required - set(manifest["files"])
    assert not missing, f"MANIFEST.json is missing bootstrap files: {sorted(missing)}"


def test_manifest_files_exist_on_disk_and_hash_matches(vendor_dir: Path, manifest: dict) -> None:
    """Every entry in MANIFEST.json must exist on disk with the recorded hash.

    This is the offline equivalent of ``vendor_pyodide.py --verify-only``
    and is what catches a vendored file being silently corrupted or a
    partial commit shipping a hash that no longer matches the bytes.
    """
    problems: list[str] = []
    for name, entry in manifest["files"].items():
        path = vendor_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
            continue
        size = path.stat().st_size
        if size != entry["size"]:
            problems.append(f"size mismatch for {name}: on-disk={size}, manifest={entry['size']}")
            continue
        h = hashlib.sha384()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = base64.b64encode(h.digest()).decode("ascii")
        if got != entry["sha384_b64"]:
            problems.append(f"hash mismatch for {name}")
    assert not problems, "MANIFEST verification failed:\n  " + "\n  ".join(problems)


def test_manifest_closure_matches_lockfile_dependencies(vendor_dir: Path, manifest: dict) -> None:
    """The closure in MANIFEST.json must be the transitive deps of the roots
    according to the vendored pyodide-lock.json.

    If a future Pyodide bump adds a new dependency for matplotlib/pandas
    and someone forgets to re-run the vendor script, this test fails
    rather than producing a half-vendored browser tool that silently
    falls back to fetching the missing dep from a CDN.
    """
    lockfile = json.loads((vendor_dir / "pyodide-lock.json").read_text(encoding="utf-8"))
    packages = lockfile["packages"]

    roots = manifest["packages_root"]
    seen: set[str] = set()
    order: list[str] = []
    stack: list[str] = list(roots)
    while stack:
        name = stack.pop(0)
        if name in seen:
            continue
        seen.add(name)
        order.append(name)
        stack.extend(packages[name].get("depends", ()))

    assert sorted(manifest["closure"]) == sorted(order), (
        "Manifest closure drifted from pyodide-lock.json deps.\n"
        f"  manifest:        {sorted(manifest['closure'])}\n"
        f"  recomputed:      {sorted(order)}\n"
        "Re-run scripts/vendor_pyodide.py to bring them back in sync."
    )

    # Each package in the closure must have its wheel vendored.
    for pkg in order:
        wheel = packages[pkg]["file_name"]
        assert wheel in manifest["files"], (
            f"Package {pkg!r} is in the closure but its wheel "
            f"{wheel!r} is not vendored. Run scripts/vendor_pyodide.py."
        )
