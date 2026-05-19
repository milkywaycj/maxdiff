# maxdiff

Open tools for designing and analyzing MaxDiff (best-worst scaling) studies.
The same analysis code ships in three forms: a zero-install browser tool
hosted on GitHub Pages, a single-file Windows desktop EXE with optional
Hierarchical Bayes, and a pip-installable Python package (`pip install
maxdiff`) for use in scripts and notebooks. Uploaded data never leaves
your machine in any of the three modes.

## Quick start

### Browser

No install. Open the tool in any modern browser; the Python analysis
runs locally under Pyodide/WebAssembly and your data is never uploaded.

- **Analysis tool**: <https://milkywaycj.github.io/maxdiff/analysis/> —
  count-based MaxDiff scores with optional bootstrap confidence
  intervals.
- **Design tool**: <https://milkywaycj.github.io/maxdiff/design/> —
  balanced MaxDiff design generation.

Hierarchical Bayes is intentionally not offered in the browser tool;
the JAX/NumPyro stack is too heavy for Pyodide.

### Desktop (Windows)

Download the latest Windows EXE from the
[releases page](https://github.com/milkywaycj/maxdiff/releases/latest).
Hierarchical Bayes is bundled — no extra install needed.

**First run on Windows.** The EXE is not code-signed (this is a free
open-source project and Microsoft does not offer a free signing
option), so Windows SmartScreen will warn you twice: once during the
download in your browser, and again when you launch the EXE. Both
warnings are expected for unsigned open-source binaries and don't
mean the EXE is unsafe — the source is in this repository and CI
builds it from a tagged commit.

1. **In the browser.** Edge (and other Chromium-based browsers) show
   a "Make sure you trust ... before you open it" dialog in the
   Downloads panel with **Cancel** and **Delete** buttons. Click the
   small chevron next to **Delete** and choose **Keep anyway**. (If
   the dialog also offers **Report this app as safe**, that's
   optional — it just submits a reputation hint to Microsoft.)
2. **At launch.** Double-clicking the EXE produces a blue "Windows
   protected your PC" SmartScreen dialog. Click the small **More
   info** link, then the **Run anyway** button that appears.

SmartScreen remembers the decision after the first run. The EXE
launches a customtkinter GUI; on launch it probes for `jax` /
`numpyro` and enables the Hierarchical Bayes option if they are
importable. The bundled EXE always has them. If you'd rather avoid
the dialogs entirely, the `pip install maxdiff` path below works on
any platform with Python and has no SmartScreen interaction.

### Python package

```bash
pip install maxdiff             # core (pandas / numpy / matplotlib)
pip install maxdiff[hb]         # + jax + numpyro for Hierarchical Bayes
pip install maxdiff[desktop]    # + customtkinter / seaborn / openpyxl
```

A minimal end-to-end use of the count analysis:

```python
import pandas as pd
from maxdiff import calculate_observed_percentages

df = pd.DataFrame({
    "Response ID":   [1, 1, 1, 2, 2, 2],
    "Attribute1":    ["A", "B", "C", "A", "B", "C"],
    "Attribute2":    ["B", "C", "A", "B", "C", "A"],
    "Attribute3":    ["C", "A", "B", "C", "A", "B"],
    "Best":          ["A", "B", "A", "B", "C", "A"],
    "Worst":         ["C", "A", "C", "A", "A", "B"],
})
scores = calculate_observed_percentages(
    df,
    attribute_columns=["Attribute1", "Attribute2", "Attribute3"],
    pos_col="Best",
    neg_col="Worst",
    output_terms=("Best", "Worst"),
)
print(scores)
```

The full public surface (count, bootstrap, display statistics,
correlation, format detection, color parsing, encoding-aware file
reads, plotting, and Hierarchical Bayes) is re-exported from the
top-level `maxdiff` namespace; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the per-module breakdown.

## Features

- **Count-based scoring.** Per-item best-rate, worst-rate, percent-
  unselected, and net score, sorted. Research consistently shows this
  correlates r > 0.99 with Hierarchical Bayes utilities on standard
  balanced designs.
- **Bootstrap confidence intervals.** Respondent-level resampling
  (preserves within-respondent task clustering), reports the observed
  score together with the 2.5 / 97.5 percentile interval.
- **Hierarchical Bayes** (optional, requires `jax` + `numpyro`). NUTS
  sampling with a non-centered respondent parameterization and a hard
  **sum-to-zero** identification constraint so the prior is symmetric
  across items and extreme items aren't biased outward. R-hat
  thresholds follow Vehtari et al. 2021.
- **Display statistics and balance metrics.** Per-item display counts,
  selection counts, and study-level balance diagnostics.
- **Segment analysis.** Run the full pipeline within a categorical
  segmentation column.
- **Correlation matrix.** Item-item Pearson correlation derived from
  best/worst selection patterns.
- **Design generation** (browser tool). Balanced MaxDiff designs with
  uniform per-version slot randomization, downloadable as CSV.
- **Format-tolerant I/O.** Encoding fallbacks (UTF-8, UTF-8-sig,
  UTF-16, CP1252, Latin-1) for the messy CSVs that come out of
  Qualtrics / SurveyMonkey / Excel; format auto-detection for
  ready-to-analyze and Qualtrics-wide layouts.
- **Thread-safe plotting.** Matplotlib OO API only (no pyplot globals,
  no seaborn) so the GUI can render plots from a worker thread without
  corrupting state.

## Project structure

```
maxdiff/
  src/
    maxdiff/                       Installable analysis package
      __init__.py                  Public API re-exports
      _version.py                  Single source of version truth
      count.py                     Count-based scoring
      bootstrap.py                 Bootstrap CIs (respondent-level)
      correlation.py               Item-item correlation matrix
      display.py                   Per-item stats and balance metrics
      formats.py                   Format detection, conversion, validation
      colors.py                    Hex / named color input parsing
      hb.py                        Hierarchical Bayes (optional jax/numpyro)
      io.py                        Encoding-aware tabular file reader
      plotting.py                  Thread-safe matplotlib (OO API) plots
    MaxDiff_Data_Analyzer_v2-1.py  customtkinter desktop GUI (imports maxdiff)

  docs/                            GitHub Pages site
    index.html                     Landing page
    analysis/index.html            Browser analysis tool (Pyodide + micropip)
    design/index.html              Browser design tool
    wheels/<version>/              In-repo wheel served to Pyodide
    vendor/pyodide-<version>/      Vendored Pyodide runtime (same-origin)
    RELEASE.md                     Release procedure
    ARCHITECTURE.md                Architecture and design notes

  scripts/
    vendor_pyodide.py              Re-runnable fetch of docs/vendor/pyodide-*

  tests/
    unit/                          Fast, isolated module tests
    property/                      Hypothesis property-based tests
    statistical/                   Monte Carlo recovery / coverage tests
    golden/                        Pinned-output regression tests
    e2e/                           Playwright browser-driven tests
    helpers/                       Synthetic data generator, loaders

  pyinstaller/                     Windows EXE build notes
  .github/workflows/               CI (Linux+Windows × Py 3.10-3.12), release
  pyproject.toml                   Build, deps, pytest, ruff, coverage config
```

## Development

```bash
git clone https://github.com/milkywaycj/maxdiff
cd maxdiff
python -m venv .venv
. .venv/bin/activate              # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .                  # editable install of the maxdiff package
```

Optional extras:

```bash
pip install -r requirements-hb.txt          # JAX + NumPyro for HB tests
playwright install chromium                  # required for tests/e2e/
```

Run the suite:

```bash
pytest -q                          # full suite, ~1-2 min with HB + e2e
pytest -q -m "not slow"            # sub-minute fast tier
pytest tests/unit                  # one tier
pytest --cov                       # with coverage
```

Lint and format checks:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

Browser end-to-end tests use Playwright + Chromium against a local
`http.server` serving `docs/`; they're in `tests/e2e/`. They skip
gracefully if Chromium isn't installed.

CI runs the matrix Linux + Windows × Python 3.10 / 3.11 / 3.12 on
every push; HB-dependent tests are installed and run on Linux only
because installing JAX on Windows runners is prohibitively slow and
the HB code paths are platform-independent.

## Releasing

See [`docs/RELEASE.md`](docs/RELEASE.md). The short version: bump
`src/maxdiff/_version.py`, update `CHANGELOG.md`, push a `v*.*.*`
tag; `.github/workflows/release.yml` builds the wheel, sdist, and
HB-enabled Windows EXE and attaches them to a GitHub Release.

## Contributing

Issues and PRs welcome at
<https://github.com/milkywaycj/maxdiff/issues>. A few norms:

- **Every behavior change ships with a test.** Test-first (red → green
  → refactor) is the working norm here, not an aspiration. New
  numerical output gets a golden; new algorithmic behavior gets a
  statistical test; new validation paths get unit tests for each error
  branch. The test pyramid (unit / property / statistical / golden /
  integration / e2e) is described in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Commit messages** are descriptive and self-contained — title
  summarizing the change in the imperative, body explaining the *why*,
  one logical change per commit. Phases of work get a `Phase N:`
  prefix when they belong to a multi-commit effort.
- **Lint and format** must pass (`ruff check`, `ruff format --check`).
- **CI must be green** on Linux and Windows across Python 3.10–3.12
  before merge.
- For larger changes, open an issue first to agree on scope. Bug
  reports with a minimal reproducing dataset (even synthetic) are
  significantly easier to action than free-form descriptions.

## License

MIT. See [LICENSE](LICENSE).
