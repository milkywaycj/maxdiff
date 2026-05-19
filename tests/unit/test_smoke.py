"""Phase 1 smoke test.

The MaxDiff analyzer currently lives in a single file with a hyphen
in its name (``src/MaxDiff_Data_Analyzer_v2-1.py``), which makes it
non-importable as a regular Python module. The test helper
``tests.helpers.legacy_loader`` loads it from path so that the rest
of the test suite can exercise its functions directly.

This module is the first test to run in the suite. Its sole job is
to assert that the harness itself is working end-to-end:

  1. The legacy loader can locate and import the analyzer.
  2. A representative set of public analysis functions are exposed.

If this test fails, no other test in the suite is meaningful. CI
should treat a failure here as a build infrastructure problem, not
a regression in the analysis code.
"""

from __future__ import annotations

from tests.helpers import legacy_loader


def test_legacy_module_imports() -> None:
    """Importing the legacy analyzer module must succeed."""
    module = legacy_loader.load_analyzer()
    assert module is not None
    assert hasattr(module, "__file__")


def test_expected_analysis_functions_exist() -> None:
    """The functions the rest of the test suite will exercise must be present.

    This test will fail loudly if a future refactor renames or removes
    one of these symbols without updating the test suite alongside.
    """
    module = legacy_loader.load_analyzer()

    expected_functions = [
        # Core count analysis
        "calculate_observed_percentages",
        "calculate_scores_no_ci",
        "perform_maxdiff_analysis",
        "bootstrap_analysis",
        # Display statistics
        "calculate_display_statistics",
        "format_display_report",
        # Correlation
        "calculate_correlation_matrix",
        # Data validation
        "check_errors",
        # Color helpers
        "process_color_input",
        # Format detection
        "detect_terminology",
        "get_column_names",
    ]

    missing = [name for name in expected_functions if not hasattr(module, name)]
    assert not missing, f"Expected functions are missing from the legacy module: {missing}"


def test_expected_classes_exist() -> None:
    """The classes the rest of the suite will exercise must be present."""
    module = legacy_loader.load_analyzer()

    expected_classes = ["DataFormatDetector", "HierarchicalBayesMaxDiff"]
    missing = [name for name in expected_classes if not hasattr(module, name)]
    assert not missing, f"Expected classes are missing from the legacy module: {missing}"
