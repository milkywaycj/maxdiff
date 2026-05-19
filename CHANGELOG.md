# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 conventions: while versions are still `< 1.0`, the public
Python API (the symbols in `maxdiff.__all__`) is allowed to change
between minor versions, but breaking changes are called out under
`### Changed` with a migration note.

## [Unreleased]

### Added
- Browser analysis tool (`docs/analysis/index.html`) now consumes the
  `maxdiff` wheel via Pyodide + micropip from `docs/wheels/<version>/`,
  eliminating roughly 570 lines of duplicated Python that previously
  lived inside the page as a `PYTHON_CODE` string. The browser and
  desktop tools now share one source of truth for the analysis math.
- **Same-origin Pyodide runtime** under
  `docs/vendor/pyodide-<version>/`. `scripts/vendor_pyodide.py`
  re-fetches Pyodide and the dependency closure of `micropip`, `numpy`,
  `pandas`, and `matplotlib` from `cdn.jsdelivr.net`, cross-verifies
  every wheel against the SHA-256 hash pinned in `pyodide-lock.json`,
  and writes a `MANIFEST.json` with per-file SHA-384 hashes. The
  browser tool now points `PYODIDE_INDEX_URL` at the vendored path, so
  every byte the user executes is served same-origin from GitHub Pages
  and is auditable in `git log`. The CDN is no longer a trust boundary
  at runtime. New tests at `tests/unit/test_pyodide_vendor.py` pin the
  invariant: no jsdelivr / unpkg / cdnjs references in the shipped
  HTML, MANIFEST.json must verify against on-disk bytes, and the
  vendored closure must match `pyodide-lock.json`'s dependency graph.

### Changed
- `tests/unit/test_pyodide_sri.py` now verifies the pinned SHA-384
  hash against the vendored `pyodide.js` instead of round-tripping to
  the CDN. The check is offline and runs in the fast tier.

### Documentation
- `docs/RELEASE.md` documents the re-vendoring procedure and replaces
  the deferred-signing options table with an explicit statement that
  the EXE will remain unsigned; `README.md` walks first-time Windows
  users through the SmartScreen dialog more carefully.

### Added (Phase 11)
- **HB regression goldens** at
  `tests/golden/test_hb_goldens.py`. A small fixed-seed HB fit
  (N=100, 500+500 MCMC iterations, 1 chain) pins the population Score
  column and the credible-interval widths to checked-in CSVs in
  `tests/golden/expected/hb_*.csv` with generous tolerances
  (`float_tol=0.1` and `0.15` respectively) sized for JAX-version
  drift, not MCMC sampling noise. Catches HB regressions that the
  recovery test (which uses a 0.5 tolerance against ground truth)
  would miss. ~9-10s, marked `hb` / `slow` / `golden`; skips when
  numpyro / jax is unavailable.

### Added (Phase 12)
- **PyPI Trusted Publishing** wired into `.github/workflows/release.yml`.
  The publish step uses OIDC (no long-lived token) and is gated on the
  GitHub repository variable `MAXDIFF_PYPI_PUBLISH` so it remains inert
  until the maintainer completes the one-time PyPI-side setup
  documented in `docs/RELEASE.md`. The `release` job now scopes
  `contents: write` and `id-token: write` per-job rather than at the
  workflow level, so the build jobs run with the default read-only
  GITHUB_TOKEN.

## [3.0.0.dev0] - 2026-05-19

This is the first formal release of the `maxdiff` Python package and
the first end-to-end versioned snapshot of the project. The previous
single-file desktop script is preserved as an entry point and is the
same numerical pipeline under the hood, but the analysis code is now
an installable library and the browser tool is a thin shell around
the same library.

### Added
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
- **Pyodide bootstrap is now SRI-verified.** `importScripts()` does
  not honor subresource integrity, so a compromised CDN response
  would previously have executed unchecked. The bootstrap is fetched
  on the main thread, its SHA-384 digest is verified via
  `crypto.subtle` against a pinned constant, and only the verified
  bytes are evaluated. WASM, stdlib, and package fetches initiated by
  Pyodide itself remain CDN-served; full vendoring is tracked as a
  future hardening item.
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
