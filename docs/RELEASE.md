# Release process

This document describes how to cut a release of `maxdiff` and the
desktop EXE. The automation lives in `.github/workflows/release.yml`
(Phase 7).

## Overview

A release produces three artifacts attached to a GitHub Release:

1. A Python **wheel** (`maxdiff-X.Y.Z-py3-none-any.whl`)
2. A Python **sdist** (`maxdiff-X.Y.Z.tar.gz`)
3. A Windows **EXE** (`MaxDiff_Data_Analyzer_vX.Y.Z.exe`, built with
   Hierarchical Bayes support included)

## Trigger modes

| Trigger | When to use | Result |
|---|---|---|
| Push a tag matching `v*.*.*` | Real release | Builds artifacts AND publishes a GitHub Release with them attached. |
| Manual `workflow_dispatch` | Testing the workflow itself | Builds artifacts and uploads them as workflow artifacts. **No Release is created.** |

## Cutting a real release

1. **Bump the version.** Edit `src/maxdiff/_version.py`:

   ```python
   __version__ = "3.0.0"     # drop the .devN suffix
   ```

   The version is the single source of truth; `pyproject.toml` reads it
   dynamically.

2. **Update `CHANGELOG.md`** (create on the first release using the
   Keep-a-Changelog format; move `Unreleased` entries under a new
   `## [3.0.0] - YYYY-MM-DD` heading).

3. **Commit and push to `main`.**

   ```bash
   git add src/maxdiff/_version.py CHANGELOG.md
   git commit -m "Release 3.0.0"
   git push origin main
   ```

4. **Tag and push the tag.**

   ```bash
   git tag v3.0.0
   git push --tags
   ```

5. **Watch the workflow.** Go to the Actions tab and wait for the
   `Release` workflow to finish. It will:
   - Build the wheel + sdist on Ubuntu and import-test them.
   - Build the Windows EXE (with HB support) on Windows.
   - Rename the EXE to include the tag (so the filename alone
     identifies the version).
   - Create a GitHub Release at
     <https://github.com/milkywaycj/maxdiff/releases/tag/v3.0.0> and
     attach all three artifacts.

6. **Post-release.** Bump `_version.py` back to a `.devN` suffix on
   `main` (e.g. `3.1.0.dev0`) so subsequent commits are clearly
   development builds.

## Testing the workflow without a real release

Use the manual trigger:

1. Open the Actions tab on GitHub.
2. Select the **Release** workflow.
3. Click **Run workflow**, optionally giving a label via the `tag`
   input (defaults to `preview`).
4. The workflow builds both artifacts and uploads them as workflow
   artifacts. The publish step is gated on `refs/tags/v*` and is
   skipped, so no Release is created.

You can download the artifacts from the run summary to verify the
wheel installs and the EXE launches on a real Windows machine.

## Required secrets

None at the moment. The `GITHUB_TOKEN` provided automatically by
Actions is sufficient to create Releases.

If you later decide to publish to PyPI, add:

- `PYPI_API_TOKEN` — a PyPI API token scoped to the `maxdiff` project.

The PyPI publish step is present in `release.yml` but **commented out**.
Uncomment it once the secret is in place. Prefer Trusted Publishing
(OIDC) over a long-lived token for new projects.

## Manual fallback

If automation is broken and you need to release by hand, see
`pyinstaller/build_instructions.md` for the local PyInstaller command.
Build the wheel locally with `python -m build`, then upload everything
to a Release manually via the GitHub UI.

## Code-signing the EXE

The EXE is **not code-signed**. Every option Microsoft offers for
trusted Windows signing costs money on an ongoing basis, and this is
a no-cost open-source project — there is no billing relationship to
attach a certificate or signing service to. The standard
open-source-on-Windows posture applies: SmartScreen will warn on
first run, and the README walks users through dismissing the warning.
This decision is intentional and final until somebody other than the
maintainer wants to fund signing. Do not add signing scaffolding to
`release.yml` without that funding in place; otherwise the workflow
gains a step that depends on a secret nobody owns.

## Re-vendoring Pyodide

`scripts/vendor_pyodide.py` produces the runtime served from
`docs/vendor/pyodide-<version>/`. Re-run it whenever the pinned
`PYODIDE_VERSION` constant in `docs/analysis/index.html` changes, or
whenever the vendored files need to be regenerated from scratch.

```bash
python scripts/vendor_pyodide.py                # incremental refresh
python scripts/vendor_pyodide.py --force        # redownload every file
python scripts/vendor_pyodide.py --verify-only  # CI-style hash check
```

The script reads `PYODIDE_VERSION` from the HTML, downloads
`pyodide-lock.json`, walks the dependency closure of `micropip`,
`numpy`, `pandas`, and `matplotlib`, fetches every file in that
closure plus the four bootstrap artifacts, cross-verifies each wheel
against the SHA-256 hash Pyodide itself pins in `pyodide-lock.json`,
and writes a `MANIFEST.json` recording per-file SHA-384 hashes for
later integrity checks. When bumping `PYODIDE_VERSION`:

1. Update the constant in `docs/analysis/index.html`.
2. Run `python scripts/vendor_pyodide.py` to populate the new
   `docs/vendor/pyodide-<new-version>/` directory.
3. Update `PYODIDE_BOOTSTRAP_SHA384` in the HTML to the new
   `pyodide.js` hash (visible in the new `MANIFEST.json`).
4. Delete the old `docs/vendor/pyodide-<old-version>/` directory.
5. Run `pytest -q tests/unit/test_pyodide_vendor.py
   tests/unit/test_pyodide_sri.py` to confirm everything is in sync.

The vendored bundle is ~75 MiB. That is the price of having every byte
the browser executes auditable in `git log` instead of trusting a CDN.
