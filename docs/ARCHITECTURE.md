# Architecture

This document describes how the `maxdiff` codebase is laid out and why,
the boundaries between the three delivery modes, and the discipline we
hold the test suite to.

## Three delivery modes

The same `maxdiff` Python package powers all three forms the tool is
shipped in:

1. **Browser** — `docs/analysis/index.html` runs Python in the user's
   browser via Pyodide (CPython compiled to WebAssembly). The Pyodide
   runtime is vendored under `docs/vendor/pyodide-<version>/` and the
   `maxdiff` wheel under `docs/wheels/<version>/`, both served
   same-origin from GitHub Pages. The page calls micropip on the
   vendored wheel and the public API directly. No data leaves the
   browser, and no third-party origin is contacted at runtime.
2. **Desktop (Windows)** — `src/MaxDiff_Data_Analyzer_v2-1.py` is a
   customtkinter GUI that imports from `maxdiff` and presents the same
   pipeline plus interactive plotting, segmentation, and Hierarchical
   Bayes. PyInstaller bundles the script, Python, the package, and
   `jax` / `numpyro` into a single EXE.
3. **Python package** — `pip install maxdiff` exposes the analysis
   primitives for scripts and notebooks. The browser tool and the
   desktop tool are both consumers of this package.

The package itself never imports tkinter, customtkinter, pyplot, or
seaborn, and it never opens a window. That constraint is what lets the
same code run in Pyodide. The GUI layer lives outside the package, in
`src/MaxDiff_Data_Analyzer_v2-1.py`.

## Package layout

All modules live in `src/maxdiff/`. Every submodule is import-safe in
WebAssembly (no native extensions beyond what Pyodide ships, no GUI).

### `__init__.py`
Public API re-exports. The complete surface in `maxdiff.__all__`:

- Count analysis: `calculate_observed_percentages`,
  `calculate_scores_no_ci`, `perform_maxdiff_analysis`
- Bootstrap: `bootstrap_analysis`
- Display statistics: `calculate_display_statistics`,
  `format_display_report`
- Correlation: `calculate_correlation_matrix`
- Format detection / validation: `DataFormatDetector`,
  `detect_terminology`, `get_column_names`, `check_errors`
- Color parsing: `process_color_input`
- File I/O: `read_tabular_file`
- Hierarchical Bayes (optional): `HierarchicalBayesMaxDiff`,
  `HAS_NUMPYRO`, `NUMPYRO_ERROR`
- Version: `__version__`

Consumers should import from the top-level `maxdiff` namespace, not
from submodules; submodule paths may change without notice.

### `_version.py`
Single source of version truth. `__version__ = "3.0.0"`. Read
dynamically by `pyproject.toml` (`tool.setuptools.dynamic.version`)
and re-exported from the package root. Kept in its own module so the
build backend can read it without importing the analysis stack.

### `count.py`
Core count-based scoring. Three functions:

- `calculate_observed_percentages` — per-item best/worst/unselected
  percentages and net score from a long-format DataFrame.
- `calculate_scores_no_ci` — same net score, score column only.
- `perform_maxdiff_analysis` — numpy-array core consumed by
  `bootstrap.py` for speed; operates on pre-encoded integer arrays.

Imported by `bootstrap.py`, by `MaxDiff_Data_Analyzer_v2-1.py`, and by
the browser tool via the wheel.

### `bootstrap.py`
Respondent-level resampling bootstrap. `bootstrap_analysis` resamples
respondent IDs with replacement (preserving within-respondent task
clustering), recomputes the net score on each resample, and reports
the 2.5 / 97.5 percentile interval alongside the observed score on
the original sample. Imports `perform_maxdiff_analysis` from `count`.

### `correlation.py`
`calculate_correlation_matrix` accumulates +1 for each best pick and
-1 for each worst pick per respondent×item, then computes the Pearson
correlation across respondents. Items chosen together as best (or as
worst) by the same respondents end up positively correlated.

### `display.py`
Per-item display counts, selection counts, best/worst rates, plus
study-level balance metrics. `format_display_report` renders the
result as the human-readable text block the desktop log pane and the
browser tool's log share.

### `formats.py`
Format detection, conversion, and input validation. `DataFormatDetector`
classifies an input as ready-to-analyze, Qualtrics wide, or unknown;
`detect_terminology` distinguishes Best/Worst from Most/Least;
`get_column_names` returns the actual case-preserved column names;
`check_errors` is the validation gate the GUI runs before invoking
the analysis pipeline. Depends only on pandas and stdlib.

### `colors.py`
`process_color_input` parses a CSS4 named color, `#RRGGBB`, `#RGB`,
`#RRGGBBAA`, or a 6-char hex without leading `#` and returns a
canonical matplotlib-compatible string, or `None` for unrecognized
input. The hex character set and length are validated against a
compiled regex.

### `hb.py`
Hierarchical Bayes MaxDiff via NumPyro NUTS. Optional dependency on
`jax` + `numpyro`; when those imports fail, `HAS_NUMPYRO` is `False`,
`NUMPYRO_ERROR` holds the message, and constructing
`HierarchicalBayesMaxDiff` raises a clear `ImportError`. The
likelihood is the Displayr/Sawtooth-style "tricked logit" (best and
worst contribute two independent multinomial choices per task);
respondent utilities use a non-centered parameterization. The
identification constraint is a hard per-respondent sum-to-zero: n-1
free utilities are sampled, the n-th is the negative sum. Population
mean is concatenated the same way. R-hat thresholds follow
Vehtari et al. 2021.

### `io.py`
`read_tabular_file` reads a CSV or XLSX path with encoding fallback:
UTF-8-sig → UTF-8 → UTF-16 → CP1252 → Latin-1, with an
`encoding_errors="replace"` last resort so the user gets parseable
output even on truly broken bytes. XLSX paths route to
`pandas.read_excel`. Accepts `str` or `pathlib.Path`.

### `plotting.py`
Thread-safe plotting. Uses matplotlib's OO API exclusively: figures
are constructed as `Figure(figsize=...)`, a `FigureCanvasAgg` is
attached, axes come from `fig.add_subplot`, and figures are saved
with `fig.savefig`. No pyplot globals. No seaborn. The correlation
heatmap uses `ax.imshow` + an attached colorbar. The desktop GUI
runs analysis on a worker thread; this module is the reason that
works without corrupting matplotlib's thread-unsafe global state.
`save_plot` and `save_dataframe` are also exported.

## Browser tool

`docs/analysis/index.html` is a single-page Pyodide application:

1. **Same-origin Pyodide.** The Pyodide runtime is vendored under
   `docs/vendor/pyodide-<version>/` by `scripts/vendor_pyodide.py`
   and served same-origin from GitHub Pages. The closure of
   `micropip`, `numpy`, `pandas`, and `matplotlib` (computed from
   Pyodide's own `pyodide-lock.json`) is committed alongside the
   core bootstrap files. Every byte the user runs is in `git log`,
   and the GitHub Pages TLS boundary covers the whole stack rather
   than just the entry-point HTML. As defense-in-depth against
   tampering with the vendored copy itself, the page still SHA-384-
   checks `pyodide.js` against a pinned constant before `eval`'ing
   it inside the worker and refuses to proceed on mismatch. The
   verified bytes are inlined into the worker blob (not loaded via
   `importScripts()`, which doesn't honor subresource integrity).
2. **`maxdiff` wheel served from the repo.** `docs/wheels/<version>/
   maxdiff-<version>-py3-none-any.whl` is installed via micropip:
   `await micropip.install(MAXDIFF_WHEEL_URL)`. Hosting the wheel
   in-repo keeps the browser tool's analysis math in lockstep with
   the desktop tool; both load the same wheel artifact. The
   `_version.py` constant lives in the wheel, so a version mismatch
   between page and wheel surfaces immediately.
3. **CSV parser and writer.** Pure JS, RFC-4180-conforming on
   output (CRLF, `""` escaping, quoting of cells containing
   `,` / `"` / CR / LF), strict on input (unclosed quoted fields
   raise instead of producing silent corruption).
4. **Rendering uses DOM APIs.** `createElement` / `textContent` /
   `replaceChildren` everywhere; no `innerHTML` of user-supplied data.
   Integer-valued numeric columns render with `toLocaleString()`,
   fractional columns with `toFixed(2)`.

The browser tool does not offer Hierarchical Bayes. The JAX/NumPyro
stack is too heavy for Pyodide and the user expectation for a browser
page is "results in seconds", not minutes.

## Desktop tool

`src/MaxDiff_Data_Analyzer_v2-1.py` is the Windows desktop entry point.
After the Phase 3 extraction it consists of:

- customtkinter GUI classes: `ColorButton`, `CollapsibleFrame`,
  `DataPreviewWindow`, `ColumnMapperWindow`, `MaxDiffGUI`.
- Segment-level orchestration: `segment_maxdiff_analysis`,
  `process_segment_results`.
- Thin glue calling into `maxdiff` for everything numerical.

All pure analysis is imported from `maxdiff`; the file is
~1495 lines and contains no scoring math. The GUI runs analysis
(including plotting) from a worker thread to keep the UI responsive,
which is why `maxdiff.plotting` is built strictly on the OO API.

PyInstaller bundles the script into a single Windows EXE. The build
command (in `pyinstaller/build_instructions.md` and reproduced in
`.github/workflows/release.yml`) passes `--onefile` and the
`jax` / `jaxlib` / `numpyro` hidden imports so the EXE ships with HB
support baked in. The EXE is renamed at release time to include the
version tag.

## Test pyramid

The test suite is structured as a pyramid; every behavior change is
expected to ship with the tier(s) appropriate to its risk profile.
Discovery is rooted at `tests/`; markers are declared in
`pyproject.toml`.

- **Unit** (`tests/unit/`). Fast, isolated, one module at a time. The
  bulk of the suite. Example: `tests/unit/test_count_analysis.py`
  hand-builds a small DataFrame with hand-counted display/best/worst
  tallies documented in a table at the top of the module and asserts
  exact arithmetic against the count-analysis functions.
- **Property** (`tests/property/`). Hypothesis sweeps across
  parametric spaces, looking for inputs that violate stated
  invariants. Example: `tests/property/test_synthetic_data_invariants.py`
  generates random `(n_respondents, n_items, items_per_task, repeats)`
  configurations and asserts that the synthetic data generator always
  produces balanced designs without within-task duplicates.
- **Statistical** (`tests/statistical/`, marker `statistical`). Monte
  Carlo correctness checks against known synthetic truth. Example:
  `tests/statistical/test_bootstrap_coverage.py` draws 80 independent
  N=200 samples from a known utility vector and asserts that the
  bootstrap CIs achieve 80–100% coverage of the long-run score
  computed on a 4000-respondent reference dataset.
- **Golden** (`tests/golden/`, marker `golden`). Pinned-output
  regression tests on small deterministic synthetic datasets. Example:
  `tests/golden/test_count_analysis_goldens.py` runs the four primary
  analysis functions on two fixed datasets and compares against
  checked-in CSVs in `tests/golden/expected/`. A `--update-goldens`
  CLI flag overwrites the pins; intentional numerical changes require
  an explicit audit trail. Missing golden files fail the test.
- **Integration**. Multi-module workflow tests (marker
  `integration`). Currently sparse; integration coverage is mostly
  achieved indirectly via the golden tier and the e2e tier.
- **End-to-end** (`tests/e2e/`, marker `e2e`). Playwright + Chromium
  driving the actual HTML in `docs/analysis/` and `docs/design/`
  against a local `http.server`. Example:
  `tests/e2e/test_render_and_tabs.py` injects a CSV cell containing
  `<script>...</script>` and asserts the script does not execute when
  the result table renders — the regression guard for the XSS fix.

Markers and tiers can be mixed: `pytest -q -m "not slow"` runs every
tier minus the long-running statistical and HB tests for sub-minute
feedback; `pytest -q -m hb` requires `jax` + `numpyro` and skips
gracefully when unavailable.

**Discipline.** Every behavior change ships with a test, on whichever
tier matches the change:

- New numerical output → golden.
- New algorithmic guarantee → statistical / property.
- New error path → unit test per branch.
- New browser-visible behavior → e2e.

The default warning filter in `pyproject.toml` is `error`; any new
warning surface becomes a test failure, forcing a deliberate decision
rather than silent rot.

## Versioning

The project follows [Semantic Versioning](https://semver.org/). The
single source of truth for the version is
`src/maxdiff/_version.py`; `pyproject.toml` reads it dynamically via
`tool.setuptools.dynamic.version`, and the top-level `maxdiff`
package re-exports it as `maxdiff.__version__`.

While the project is at `< 1.0`, the public Python API
(`maxdiff.__all__`) may change between minor versions; breaking
changes are called out under `### Changed` in `CHANGELOG.md` with a
migration note. After `1.0`, standard SemVer applies.

The release flow is:

1. Bump `src/maxdiff/_version.py` to a non-`.devN` version on `main`.
2. Move `Unreleased` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
3. Commit, push, then push a `vX.Y.Z` tag.
4. `.github/workflows/release.yml` builds the wheel + sdist + Windows
   EXE on tag push and attaches them to a GitHub Release.
5. Post-release, bump `_version.py` back to the next `.devN` so
   subsequent commits are clearly development builds.

See `docs/RELEASE.md` for the full procedure including the
`workflow_dispatch` dry-run trigger.
