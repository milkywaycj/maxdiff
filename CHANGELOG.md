# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 conventions: while versions are still `< 1.0`, the public
Python API (the symbols in `maxdiff.__all__`) is allowed to change
between minor versions, but breaking changes are called out under
`### Changed` with a migration note.

## [Unreleased]

## [3.0.3] - 2026-05-21

No changes to the analysis code or any of the three delivery modes.
Documentation, packaging metadata, and CI maintenance only.

### Documentation
- The GitHub Pages landing page (`docs/index.html`) gained a
  **Python Package** card (`pip install maxdiff`, links to PyPI) and
  a **Further Reading** card linking the Quirk's article
  "MaxDiff's max myths" on when MaxDiff is — and isn't — the right
  tool for a study.

### Packaging
- Broadened the PyPI keyword list with `best-worst-scaling`, `bws`,
  `discrete-choice`, `conjoint`, and `market-research` so PyPI search
  surfaces the project under the academic "Best-Worst Scaling"
  naming as well as the practitioner "MaxDiff" term. No rename — the
  project name stays `maxdiff`.

### CI / Infrastructure
- Bumped `actions/checkout` to v5 and `actions/setup-python` to v6
  across both workflows, off the deprecated Node 20 runtime ahead of
  GitHub's 2026-06-02 forced migration to Node 24.

## [3.0.2] - 2026-05-21

No changes to the analysis code or any of the three delivery modes.
The entries below are CI / test-infrastructure hardening only. This
is the first release published to PyPI (via Trusted Publishing — see
`docs/RELEASE.md`); `pip install maxdiff` resolves from PyPI as of
this version.

### CI / Infrastructure
- The CI matrix installs the Playwright Chromium browser
  (`playwright install --with-deps chromium`) before running tests.
  The workflow installed the `pytest-playwright` plugin but not the
  browser, so the `tests/e2e/` fixtures errored at setup on every
  matrix job.
- `jax` and `numpyro` are pinned to known-good ranges
  (`jax>=0.4,<0.5`, `numpyro>=0.13,<0.20`) in `requirements-hb.txt`.
  The HB goldens reproduce within tolerance only against the JAX
  release family they were generated on; an unpinned range would let
  the goldens drift on every JAX upgrade. (Dev/test requirement only;
  the `maxdiff[hb]` extra in `pyproject.toml` is unaffected.)
- The e2e test server (`tests/e2e/conftest.py`) binds directly to
  port 0 instead of probing for a free port and re-binding. The
  probe-then-rebind pattern was a TOCTOU race that could surface as
  intermittent `EADDRINUSE` failures.
- CI runs pytest serially (dropped `-n auto`) and emits a per-job
  JUnit XML uploaded as an artifact even on failure, so CI failures
  are diagnosable without authenticated log access.
- The matplotlib "glyph missing" `UserWarning` filter is broadened
  to match both the historical wording (`missing from current font`)
  and the current one (`missing from font(s) <name>`). The strict
  `filterwarnings = error` policy had been elevating this cosmetic
  warning to a test failure on runners that resolved a newer
  matplotlib.

## [3.0.1] - 2026-05-19

### Fixed
- **HB sum-to-zero parameterization is now actually symmetric.** v3.0.0
  shipped a sum-to-zero scheme that derived the last item's utility as
  the negative sum of the other ``n - 1``, inflating that one item's
  posterior credible interval by roughly ``sqrt(n - 1)`` relative to
  its neighbors. Point estimates were unaffected (the recovery test and
  per-item CI golden — generated *with* the bug in place — both
  passed), but analyst-visible HB plots showed one item with an
  anomalously wide CI on real n=20 datasets. v3.0.1 replaces the
  parameterization with Stan's ``sum_to_zero_vector`` /
  PyMC's ``ZeroSumNormal`` construction: an orthonormal basis ``Q`` of
  the sum-to-zero subspace, with the ``n - 1`` free parameters sampled
  in that basis and mapped to ``n`` item-space utilities via ``Q``. The
  implied prior is now identical across all items. See
  ``src/maxdiff/hb.py::_sum_to_zero_basis`` and ``_model``.

### Added
- **Regression test for HB CI-width symmetry** at
  ``tests/golden/test_hb_goldens.py::
  test_hb_ci_widths_are_symmetric_across_items``. Asserts the
  max-to-min ratio of per-item CI widths stays below 1.5x on a balanced
  n=12 fixture, comfortably above expected MC jitter (~1.0-1.3x) and
  below the buggy ratio (~1.6-1.7x at this size). Catches the class of
  bug v3.0.0 shipped — point estimates correct, one item's CI inflated
  by an asymmetric prior — without requiring a particular item to be
  the singled-out one. The HB-golden CSVs are regenerated against the
  new parameterization.

## [3.0.0] - 2026-05-19

This is the first formal release of the `maxdiff` Python package and
the first end-to-end versioned snapshot of the project. The previous
single-file desktop script is preserved as an entry point and is the
same numerical pipeline under the hood, but the analysis code is now
an installable library and the browser tool is a thin shell around
the same library.

### Added
- **Same-origin Pyodide runtime** under
  `docs/vendor/pyodide-<version>/`. `scripts/vendor_pyodide.py`
  re-fetches Pyodide and the dependency closure of `micropip`, `numpy`,
  `pandas`, and `matplotlib` from `cdn.jsdelivr.net`, cross-verifies
  every wheel against the SHA-256 hash pinned in `pyodide-lock.json`,
  and writes a `MANIFEST.json` with per-file SHA-384 hashes. The
  browser tool's `PYODIDE_INDEX_URL` points at the vendored path, so
  every byte the user executes is served same-origin from GitHub Pages
  and is auditable in `git log`. The CDN is no longer a trust boundary
  at runtime. Tests at `tests/unit/test_pyodide_vendor.py` pin the
  invariant: no jsdelivr / unpkg / cdnjs references in the shipped
  HTML, the MANIFEST must verify against on-disk bytes, and the
  vendored closure must match `pyodide-lock.json`'s dependency graph.
- **HB regression goldens** at
  `tests/golden/test_hb_goldens.py`. A small fixed-seed HB fit
  (N=100, 500+500 MCMC iterations, 1 chain) pins the population Score
  column and the credible-interval widths to checked-in CSVs with
  generous tolerances (`float_tol=0.1` and `0.15`) sized for
  JAX-version drift, not MCMC sampling noise. Catches HB regressions
  that the recovery test (which uses a 0.5 tolerance against ground
  truth) would miss. ~10s, marked `hb` / `slow` / `golden`; skips when
  numpyro / jax is unavailable.
- **PyPI Trusted Publishing** wired into `.github/workflows/release.yml`.
  The publish step uses OIDC (no long-lived token) and is gated on the
  repo variable `MAXDIFF_PYPI_PUBLISH` so it remains inert until the
  maintainer completes the one-time PyPI-side setup documented in
  `docs/RELEASE.md`. The `release` job scopes `contents: write` and
  `id-token: write` per-job; build jobs stay on the default read-only
  GITHUB_TOKEN.
- **Installable Python package** at `src/maxdiff/` with submodules
  `count`, `bootstrap`, `correlation`, `display`, `formats`, `colors`,
  `hb`, `io`, `plotting`, and a public API re-exported from the
  top-level `maxdiff` namespace. `pip install maxdiff` (core),
  `maxdiff[hb]` (Hierarchical Bayes), `maxdiff[desktop]`
  (customtkinter GUI extras). See `src/maxdiff/__init__.py`.
- **Hierarchical Bayes MaxDiff estimator** (`src/maxdiff/hb.py`) using
  NumPyro NUTS with a non-centered respondent parameterization and a
  hard **sum-to-zero** identification constraint. The model recovers
  utilities within ~0.5 on N=400 across the full prior support; see
  `tests/statistical/test_hb_recovery.py`.
- **Encoding-aware tabular file reader** (`maxdiff.read_tabular_file`,
  `src/maxdiff/io.py`) tries UTF-8-sig → UTF-8 → UTF-16 → CP1252 →
  Latin-1 in order, with an `errors="replace"` last resort. Replaces
  the desktop GUI's previous bare `pd.read_csv(filename)` call that
  raised `UnicodeDecodeError` on anything that wasn't strict UTF-8.
- **Thread-safe plotting module** (`src/maxdiff/plotting.py`) using
  matplotlib's OO API exclusively (`Figure` + `FigureCanvasAgg`, no
  pyplot globals). Plots can be rendered from a worker thread without
  corrupting global state or leaking figures.
- **GitHub Actions release pipeline** at `.github/workflows/release.yml`:
  on push of a `v*.*.*` tag, builds the wheel + sdist on Ubuntu and
  the Windows EXE (with `jax` / `jaxlib` / `numpyro` hidden imports
  so HB is bundled) on Windows, and attaches all three to a GitHub
  Release. See `docs/RELEASE.md`. A `workflow_dispatch` trigger
  performs a dry run that uploads workflow artifacts without
  publishing.
- **Full test pyramid** with 170+ tests across `tests/unit/`,
  `tests/property/` (Hypothesis), `tests/statistical/` (Monte Carlo
  coverage + recovery), `tests/golden/` (pinned outputs),
  `tests/integration/`, and `tests/e2e/` (Playwright). Custom pytest
  markers: `slow`, `hb`, `statistical`, `golden`, `integration`,
  `e2e`.
- **CI matrix** (`.github/workflows/ci.yml`) covering Linux + Windows
  × Python 3.10 / 3.11 / 3.12 with ruff lint+format, pytest under
  coverage parallelized via `pytest-xdist`, and HB-dependent tests
  installed only on Linux.
- **Synthetic data generator** at `tests/helpers/synthetic_data.py`
  for property-based and statistical tests: simulates the standard
  Gumbel-shocked best/worst MaxDiff response model from a known
  utility vector with full input validation.
- **Subresource integrity verification** for the Pyodide bootstrap in
  the browser tool. The bootstrap script is fetched from the CDN on
  the main thread, its SHA-384 digest verified against a pinned
  constant, and the verified bytes are inlined into the worker blob
  before evaluation.
- **Browser CSV parser and writer** that handle `""` escaping,
  CRLF / LF / CR line endings, embedded newlines inside quoted
  fields, RFC-4180-conforming output, and explicit unclosed-quote
  rejection (in place of silent corruption). Tested under Playwright
  in `tests/e2e/test_csv_parser.py`.
- **First-class developer docs**: this `CHANGELOG.md`, the
  expanded top-level `README.md`, `docs/ARCHITECTURE.md`,
  `docs/RELEASE.md`, `tests/golden/README.md`,
  `pyinstaller/build_instructions.md`.

### Changed
- **HB identification constraint switched from "last item fixed at 0"
  to a hard per-respondent sum-to-zero.** The previous parameterization
  drew n-1 items from `Normal(0, 2)` with the n-th item arbitrarily
  pinned at 0; "last" was set by `pd.unique(...)` first-occurrence
  order and was effectively random, and the asymmetric prior pulled
  extreme items outward when the posterior was zero-centered in
  post-processing. The new model samples n-1 free utilities and
  derives the n-th as their negative sum, so all items share the
  same prior. See `src/maxdiff/hb.py` and the recovery tests in
  `tests/statistical/test_hb_recovery.py`.
- **Browser analysis tool now loads `maxdiff` via Pyodide+micropip**
  instead of inlining the Python pipeline as a string. Eliminates the
  drift hazard between the browser tool and the desktop tool. The
  wheel is hosted in-repo at `docs/wheels/<version>/`.
- **HB R-hat thresholds tightened** to follow Vehtari et al. 2021:
  <1.01 "Excellent", <1.05 "Good", <1.10 "borderline" (advise more
  iterations), ≥1.10 "FAILED". Replaces the previous laxer 1.05–1.10
  "Good" / 1.10–1.20 "Acceptable" scheme.
- **Subprocess calls** in the desktop GUI's "open results folder"
  switched from string-interpolated `subprocess.Popen(f'explorer
  "{path}"')` to list form across all platform branches.
- **Browser CSV export** now quotes any cell containing comma, double
  quote, CR, or LF, coerces non-strings via `String()`, and emits CRLF
  line terminators per RFC 4180.
- **Desktop entry-point script** (`src/MaxDiff_Data_Analyzer_v2-1.py`)
  shrunk from a 2417-line monolith to ~1495 lines by importing all
  pure analysis from `maxdiff`. GUI behavior and numerical output
  unchanged, verified by the golden tests.
- **`process_color_input` validates hex strings** against
  `#RGB` / `#RRGGBB` / `#RRGGBBAA`. Previously any string starting
  with `#` was returned unchanged, so `#GGGGGG` was accepted.
- **HB `_prepare_data` raises `ValueError`** with a descriptive
  message (respondent ID, task index, problem) for malformed tasks
  instead of silently padding with item index 0. The legacy behavior
  was unreachable in the GUI because of the upstream `check_errors`
  gate but was a footgun for script users.
- **Browser tool's `showTab`** now takes an explicit `event` argument
  via `onclick="showTab(event, ...)"` and uses `event.currentTarget`,
  instead of relying on the non-standard implicit global `event`.
- **Browser tool integer formatting**: numeric columns whose values
  are integer-valued (e.g. "Times Displayed") now render with
  `toLocaleString()` (thousands separator, no decimals) rather than
  `toFixed(2)` (which produced "1234.00").
- **`perform_maxdiff_analysis` casts bincount inputs to `np.intp`**
  so it works under Pyodide's strict numpy build (which refuses to
  silently downcast int64 to int32 inside `np.bincount`). No-op on
  64-bit CPython.

### Fixed
- **Stored-XSS-on-self in the browser analysis tool**: every
  `innerHTML` interpolation of user-supplied CSV data was replaced
  with DOM APIs (`createElement` / `textContent` / `replaceChildren`).
  A CSV cell like `<img src=x onerror=alert(1)>` no longer executes
  when results render.
- **Design tool position uniformity** clarified: what looked like a
  position-balance bug was finite-sample noise on the design tool's
  internal `designs[v]` storage; the user-facing `getDesignTable()`
  output is correct. The position-balance regression test in
  `tests/e2e/test_design_tool_balance.py` now exercises
  `getDesignTable(false)` at 1000 versions where CV converges to the
  noise floor for an IID-uniform process.
- **Hypothesis-surfaced edge case in the synthetic data generator**:
  shuffle-and-retry could fail to find a valid design when
  `items_per_task == n_items`. Added a greedy-fill fallback that
  picks from the most-frequent remaining items first.

### Security
- **Pyodide and its full dependency closure are vendored same-origin**
  under `docs/vendor/pyodide-<version>/`. The CDN is no longer a trust
  boundary at runtime — the GitHub Pages TLS boundary now covers
  every byte the browser tool executes, including WASM, stdlib, and
  every wheel `micropip` loads. The pinned SHA-384 check on the
  bootstrap loader remains as defense-in-depth against tampering
  with the vendored copy itself. (Supersedes the SRI-only posture in
  earlier internal builds, which left WASM / stdlib / wheels
  CDN-served.)
- **CSV input validation in the browser tool** rejects unclosed
  quoted fields rather than producing silent data corruption.

### Removed
- **`seaborn` runtime dependency** of the analysis package. The
  correlation heatmap is rendered with `ax.imshow` + an attached
  colorbar; visually equivalent. `seaborn` remains an optional
  `[desktop]` extra for now but is not imported anywhere in
  `src/maxdiff/`.
- **The legacy "last item fixed at 0" reference parameterization** in
  HB MaxDiff. Replaced by sum-to-zero (see `### Changed`).
- **The pyplot-based plotting code** in `src/MaxDiff_Data_Analyzer_v2-1.py`
  (`plot_*`, `save_*` functions) — moved to `maxdiff.plotting` and
  rewritten on the OO API.
- **Phantom-item-0 padding** in the HB `_prepare_data` path —
  replaced by an explicit raise.
