"""Vendor Pyodide into ``docs/vendor/pyodide-<version>/``.

The browser analysis tool serves the Pyodide runtime same-origin from the
GitHub Pages site rather than fetching it from a CDN, so the TLS boundary
that protects ``docs/analysis/index.html`` also covers Pyodide and every
wheel it loads. This eliminates the CDN as a separate trust boundary and
makes the deployment auditable: every byte the user runs is in
``git log``.

The script is **re-runnable**. On a second run it computes SHA-384 of
each file on disk and skips any file whose hash already matches the
MANIFEST. Pass ``--force`` to redownload everything.

Inputs
------
Pyodide version is taken from ``docs/analysis/index.html`` (the
``PYODIDE_VERSION`` constant). Override on the command line with
``--version vX.Y.Z`` for a dry run against a different release.

Outputs
-------
``docs/vendor/pyodide-<version>/`` containing:

* The four bootstrap artifacts: ``pyodide.js``, ``pyodide.asm.js``,
  ``pyodide.asm.wasm``, ``python_stdlib.zip``.
* ``pyodide-lock.json`` (the package index Pyodide consults at runtime;
  historically called ``repodata.json``).
* Every wheel in the closure of ``micropip``, ``numpy``, ``pandas``,
  ``matplotlib`` as derived from ``pyodide-lock.json``.
* ``MANIFEST.json`` mapping ``<file_name> -> {"sha384_b64": ..., "size": ...}``.

Usage
-----
::

    python scripts/vendor_pyodide.py                # incremental
    python scripts/vendor_pyodide.py --force        # redownload all
    python scripts/vendor_pyodide.py --version v0.24.1
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS_HTML = _REPO_ROOT / "docs" / "analysis" / "index.html"
_VENDOR_ROOT = _REPO_ROOT / "docs" / "vendor"

# Packages required by docs/analysis/index.html at runtime. Their
# transitive dependencies are resolved from repodata.json.
_ROOT_PACKAGES = ("micropip", "numpy", "pandas", "matplotlib")

# Core bootstrap files that aren't enumerated in repodata.json's
# packages section but are needed to initialise Pyodide.
_BOOTSTRAP_FILES = (
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)

_VERSION_RE = re.compile(r"const PYODIDE_VERSION\s*=\s*['\"]([^'\"]+)['\"]")


# ---------------------------------------------------------------------------
# Helpers


def _read_pinned_version() -> str:
    text = _ANALYSIS_HTML.read_text(encoding="utf-8")
    m = _VERSION_RE.search(text)
    if not m:
        raise SystemExit(
            "Could not find PYODIDE_VERSION in docs/analysis/index.html. "
            "Pass --version vX.Y.Z explicitly."
        )
    return m.group(1)


def _sha384_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")


def _hash_file(path: Path) -> str:
    h = hashlib.sha384()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download {url}: {exc}") from exc


def _resolve_closure(lockfile: dict, roots: Iterable[str]) -> list[str]:
    """Return the topologically-flat list of package names required by
    ``roots`` together with all transitive dependencies."""
    packages: dict[str, dict] = lockfile["packages"]
    seen: set[str] = set()
    order: list[str] = []
    queue: deque[str] = deque(roots)
    while queue:
        name = queue.popleft()
        if name in seen:
            continue
        if name not in packages:
            raise SystemExit(
                f"Package '{name}' not found in repodata.json. Available roots: "
                f"{sorted(packages)[:10]}..."
            )
        seen.add(name)
        order.append(name)
        for dep in packages[name].get("depends", ()):
            queue.append(dep)
    return order


# ---------------------------------------------------------------------------
# Main flow


def vendor(version: str, force: bool = False) -> Path:
    """Vendor the named Pyodide release. Returns the output directory."""
    version_dir = _VENDOR_ROOT / f"pyodide-{version.lstrip('v')}"
    version_dir.mkdir(parents=True, exist_ok=True)
    cdn_root = f"https://cdn.jsdelivr.net/pyodide/{version}/full/"

    manifest_path = version_dir / "MANIFEST.json"
    existing_manifest: dict[str, dict] = {}
    if manifest_path.exists() and not force:
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        except (json.JSONDecodeError, KeyError):
            existing_manifest = {}

    # Always (re)download pyodide-lock.json first; the package closure depends on it.
    lock_path = version_dir / "pyodide-lock.json"
    print(f"Fetching pyodide-lock.json from {cdn_root} ...", flush=True)
    lock_bytes = _download(cdn_root + "pyodide-lock.json")
    lock_path.write_bytes(lock_bytes)
    lockfile = json.loads(lock_bytes)

    closure = _resolve_closure(lockfile, _ROOT_PACKAGES)
    package_index = lockfile["packages"]
    package_files = [package_index[name]["file_name"] for name in closure]

    files_to_fetch: list[str] = list(_BOOTSTRAP_FILES) + package_files
    # Deduplicate while preserving order. pyodide-lock.json appears in
    # _BOOTSTRAP_FILES; we already wrote it above and want to keep that copy.
    seen: set[str] = set()
    ordered: list[str] = []
    for f in files_to_fetch:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    files_to_fetch = ordered

    # Build a filename -> pinned sha256 lookup from the lockfile so we
    # can refuse any wheel whose bytes don't match Pyodide's own pin.
    pinned_sha256: dict[str, str] = {
        entry["file_name"]: entry["sha256"]
        for entry in package_index.values()
        if "file_name" in entry and "sha256" in entry
    }

    manifest: dict[str, dict] = {}
    for filename in files_to_fetch:
        dest = version_dir / filename
        url = cdn_root + filename
        if dest.exists() and not force:
            current_hash = _hash_file(dest)
            recorded = existing_manifest.get(filename, {}).get("sha384_b64")
            if recorded and current_hash == recorded:
                manifest[filename] = {
                    "sha384_b64": current_hash,
                    "size": dest.stat().st_size,
                }
                print(f"  skip   {filename}  ({dest.stat().st_size:>9} bytes, hash OK)", flush=True)
                continue
        print(f"  fetch  {filename} ...", flush=True)
        payload = lock_bytes if filename == "pyodide-lock.json" else _download(url)
        if filename in pinned_sha256:
            computed_sha256 = hashlib.sha256(payload).hexdigest()
            if computed_sha256 != pinned_sha256[filename]:
                raise SystemExit(
                    f"sha256 mismatch for {filename}:\n"
                    f"  expected (pyodide-lock.json): {pinned_sha256[filename]}\n"
                    f"  got (CDN bytes):              {computed_sha256}\n"
                    "Refusing to vendor a wheel that doesn't match Pyodide's own pin."
                )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        manifest[filename] = {
            "sha384_b64": _sha384_b64(payload),
            "size": len(payload),
        }
        print(f"         {len(payload):>9} bytes", flush=True)

    manifest_body = {
        "pyodide_version": version,
        "source": cdn_root,
        "packages_root": list(_ROOT_PACKAGES),
        "closure": closure,
        "files": manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = sum(entry["size"] for entry in manifest.values())
    print(
        f"\nVendored {len(manifest)} files ({total / (1024 * 1024):.1f} MiB total) into "
        f"{version_dir.relative_to(_REPO_ROOT)}",
        flush=True,
    )
    return version_dir


def verify(version_dir: Path) -> list[str]:
    """Verify on-disk files match MANIFEST.json. Returns list of mismatches."""
    manifest_path = version_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return ["MANIFEST.json missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    problems: list[str] = []
    for name, entry in manifest.items():
        path = version_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
            continue
        if _hash_file(path) != entry["sha384_b64"]:
            problems.append(f"hash mismatch: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None, help="Pyodide version (e.g. v0.24.1)")
    parser.add_argument("--force", action="store_true", help="Redownload all files")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Recompute hashes and exit non-zero on any mismatch.",
    )
    args = parser.parse_args(argv)

    version = args.version or _read_pinned_version()
    version_dir = _VENDOR_ROOT / f"pyodide-{version.lstrip('v')}"

    if args.verify_only:
        problems = verify(version_dir)
        if problems:
            print("Verification FAILED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"OK: {version_dir.relative_to(_REPO_ROOT)} matches MANIFEST.json")
        return 0

    vendor(version, force=args.force)
    problems = verify(version_dir)
    if problems:
        print("Post-vendor verification FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Post-vendor verification OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
