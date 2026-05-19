"""maxdiff - pure analysis library for MaxDiff (best-worst scaling) studies.

This package contains the analysis primitives used by both the desktop
GUI and the browser-based (Pyodide) tool. No tkinter / matplotlib /
GUI dependencies are imported here - the package is safe to use in
scripts, notebooks, and WebAssembly.

The desktop GUI lives outside this package; the browser tool will
eventually load this package via micropip once the wheel is published.

Public API (re-exports from submodules):

  * Count-based analysis
      calculate_scores_no_ci, calculate_observed_percentages,
      perform_maxdiff_analysis
  * Bootstrap
      bootstrap_analysis
  * Display statistics
      calculate_display_statistics, format_display_report
  * Correlation
      calculate_correlation_matrix
  * Format detection and conversion
      DataFormatDetector, detect_terminology, get_column_names,
      check_errors
  * Color input parsing
      process_color_input
  * Hierarchical Bayes (optional, requires jax + numpyro)
      HierarchicalBayesMaxDiff, HAS_NUMPYRO, NUMPYRO_ERROR
"""

from __future__ import annotations

from maxdiff._version import __version__
from maxdiff.bootstrap import bootstrap_analysis
from maxdiff.colors import process_color_input
from maxdiff.correlation import calculate_correlation_matrix
from maxdiff.count import (
    calculate_observed_percentages,
    calculate_scores_no_ci,
    perform_maxdiff_analysis,
)
from maxdiff.display import calculate_display_statistics, format_display_report
from maxdiff.formats import (
    DataFormatDetector,
    check_errors,
    detect_terminology,
    get_column_names,
)
from maxdiff.hb import HAS_NUMPYRO, NUMPYRO_ERROR, HierarchicalBayesMaxDiff
from maxdiff.io import read_tabular_file

__all__ = [
    "HAS_NUMPYRO",
    "NUMPYRO_ERROR",
    "DataFormatDetector",
    "HierarchicalBayesMaxDiff",
    "__version__",
    "bootstrap_analysis",
    "calculate_correlation_matrix",
    "calculate_display_statistics",
    "calculate_observed_percentages",
    "calculate_scores_no_ci",
    "check_errors",
    "detect_terminology",
    "format_display_report",
    "get_column_names",
    "perform_maxdiff_analysis",
    "process_color_input",
    "read_tabular_file",
]
