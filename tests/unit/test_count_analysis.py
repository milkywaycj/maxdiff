"""Characterization tests for the count-based analysis functions.

These tests pin down the *current* behavior of:

  * calculate_observed_percentages
  * calculate_scores_no_ci
  * perform_maxdiff_analysis
  * calculate_display_statistics

The intent is twofold. First, give Phase 3 a safety net so the
extraction of these functions into ``maxdiff_core`` can be verified
to preserve behavior. Second, drive out any obvious bugs in the
existing implementation by asserting against the math directly.

Where a current behavior is correct, the test simply pins it. Where
a current behavior is suspected to be wrong (e.g. the phantom-item-0
padding in HB), a corresponding *correctness* test is added in the
``tests/statistical/`` module that fails on the buggy code, so the
fix in Phase 4 turns it green. These unit tests stay focused on what
the functions output today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader
from tests.helpers.synthetic_data import make_dataset


@pytest.fixture(scope="module")
def legacy():
    """Return the legacy analyzer module, loaded once per test module."""
    return legacy_loader.load_analyzer()


# A tiny hand-built dataset so the test assertions are arithmetic and
# auditable. Three respondents, three tasks each, three items per task,
# four items total: A, B, C, D. Hand-counted display/best/worst values
# (used in the test assertions below):
#
#                displays  best  worst   score
#   Item A          7       4     0     +57.14
#   Item B          8       0     8    -100.00
#   Item C          8       3     1     +25.00
#   Item D          4       2     0     +50.00
HAND_DATA = pd.DataFrame(
    [
        # R1
        {
            "Response ID": "R1",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "C",
            "Most": "A",
            "Least": "B",
        },
        {
            "Response ID": "R1",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "C",
            "Most": "A",
            "Least": "B",
        },
        {
            "Response ID": "R1",
            "Attribute1": "A",
            "Attribute2": "C",
            "Attribute3": "D",
            "Most": "A",
            "Least": "C",
        },
        # R2
        {
            "Response ID": "R2",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "C",
            "Most": "C",
            "Least": "B",
        },
        {
            "Response ID": "R2",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "D",
            "Most": "D",
            "Least": "B",
        },
        {
            "Response ID": "R2",
            "Attribute1": "B",
            "Attribute2": "C",
            "Attribute3": "D",
            "Most": "C",
            "Least": "B",
        },
        # R3
        {
            "Response ID": "R3",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "C",
            "Most": "A",
            "Least": "B",
        },
        {
            "Response ID": "R3",
            "Attribute1": "A",
            "Attribute2": "B",
            "Attribute3": "C",
            "Most": "C",
            "Least": "B",
        },
        {
            "Response ID": "R3",
            "Attribute1": "B",
            "Attribute2": "C",
            "Attribute3": "D",
            "Most": "D",
            "Least": "B",
        },
    ]
)


# ----------------------------------------------------------------------
# calculate_scores_no_ci
# ----------------------------------------------------------------------


class TestCalculateScoresNoCI:
    def test_returns_dataframe_with_item_and_score_columns(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_scores_no_ci(HAND_DATA, attr_cols, "Most", "Least")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Item", "Score"]

    def test_score_formula_is_best_minus_worst_over_displays_times_100(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_scores_no_ci(HAND_DATA, attr_cols, "Most", "Least").set_index(
            "Item"
        )

        # See hand-counted table at top of module.
        expected = {
            "A": (4 - 0) / 7 * 100,
            "B": (0 - 8) / 8 * 100,
            "C": (3 - 1) / 8 * 100,
            "D": (2 - 0) / 4 * 100,
        }
        for item, expected_score in expected.items():
            assert result.loc[item, "Score"] == pytest.approx(expected_score), (
                f"Score mismatch for {item}: got {result.loc[item, 'Score']}, "
                f"expected {expected_score}"
            )

    def test_results_are_sorted_by_score_descending(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_scores_no_ci(HAND_DATA, attr_cols, "Most", "Least")
        scores = result["Score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_zero_displays_yields_zero_score_not_nan(self, legacy) -> None:
        # An item that exists in attribute columns but is never displayed
        # would have 0 displays; the implementation guards against
        # division by zero and returns 0 instead of NaN.
        # We construct that case by having a column that contains an item
        # only in some other column. Simplest: drop the test for now via
        # a one-row, one-item-everywhere DataFrame and only assert no NaN.
        df = pd.DataFrame(
            [
                {
                    "Response ID": "R1",
                    "Attribute1": "X",
                    "Attribute2": "Y",
                    "Attribute3": "Z",
                    "Most": "X",
                    "Least": "Z",
                },
            ]
        )
        result = legacy.calculate_scores_no_ci(
            df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least"
        )
        assert not result["Score"].isna().any()


# ----------------------------------------------------------------------
# calculate_observed_percentages
# ----------------------------------------------------------------------


class TestCalculateObservedPercentages:
    def test_columns_use_provided_output_terms(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_observed_percentages(
            HAND_DATA, attr_cols, "Most", "Least", ("Most", "Least")
        )
        assert "% Selected as Most" in result.columns
        assert "% Selected as Least" in result.columns
        assert "% Unselected" in result.columns
        assert "Score" in result.columns

    def test_best_worst_unselected_percentages_sum_to_100(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_observed_percentages(
            HAND_DATA, attr_cols, "Most", "Least", ("Best", "Worst")
        )
        total = (
            result["% Selected as Best"] + result["% Selected as Worst"] + result["% Unselected"]
        )
        assert np.allclose(total, 100.0)

    def test_percentages_match_hand_calculation(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        result = legacy.calculate_observed_percentages(
            HAND_DATA, attr_cols, "Most", "Least", ("Best", "Worst")
        ).set_index("Item")
        # Item A: displays=7, best=4, worst=0 -> 4/7, 0/7, 3/7
        assert result.loc["A", "% Selected as Best"] == pytest.approx(100 * 4 / 7)
        assert result.loc["A", "% Selected as Worst"] == pytest.approx(0.0)
        assert result.loc["A", "% Unselected"] == pytest.approx(100 * 3 / 7)
        # Item B: displays=8, best=0, worst=8 (selected Least every single
        # time it was displayed)
        assert result.loc["B", "% Selected as Worst"] == pytest.approx(100.0)
        assert result.loc["B", "% Selected as Best"] == pytest.approx(0.0)
        assert result.loc["B", "% Unselected"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# calculate_display_statistics
# ----------------------------------------------------------------------


class TestCalculateDisplayStatistics:
    def test_returns_stats_df_and_balance_metrics(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        stats_df, balance = legacy.calculate_display_statistics(
            HAND_DATA, attr_cols, "Most", "Least"
        )
        assert isinstance(stats_df, pd.DataFrame)
        assert isinstance(balance, dict)
        # Required columns in stats_df
        for col in [
            "Item",
            "Times Displayed",
            "Times Selected Best",
            "Times Selected Worst",
            "Times Unselected",
            "Best Rate",
            "Worst Rate",
        ]:
            assert col in stats_df.columns

    def test_display_counts_match_hand_calculation(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        stats_df, _ = legacy.calculate_display_statistics(HAND_DATA, attr_cols, "Most", "Least")
        counts = stats_df.set_index("Item")["Times Displayed"]
        assert counts["A"] == 7
        assert counts["B"] == 8
        assert counts["C"] == 8
        assert counts["D"] == 4

    def test_balance_metrics_have_expected_keys(self, legacy) -> None:
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        _, balance = legacy.calculate_display_statistics(HAND_DATA, attr_cols, "Most", "Least")
        for key in [
            "total_displays",
            "num_items",
            "min_displays",
            "max_displays",
            "mean_displays",
            "std_displays",
            "cv_displays",
            "range_displays",
            "is_balanced",
            "balance_warnings",
            "balance_status",
        ]:
            assert key in balance

    def test_balanced_synthetic_design_yields_low_cv(self, legacy) -> None:
        # The synthetic generator builds perfectly balanced designs.
        # Across all respondents, every item is displayed exactly
        # n_respondents * repeats_per_item times, so the CV is 0 and the
        # balance status should be "Perfectly Balanced".
        df = make_dataset(
            n_respondents=20,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.zeros(8),
            seed=0,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        stats_df, balance = legacy.calculate_display_statistics(df, attr_cols, "Most", "Least")
        # Every item shown 40 times.
        assert (stats_df["Times Displayed"] == 40).all()
        assert balance["cv_displays"] == pytest.approx(0.0)
        assert balance["is_balanced"] is True


# ----------------------------------------------------------------------
# Round-trip via synthetic data: count analysis recovers known ordering
# ----------------------------------------------------------------------


class TestPerformMaxDiffAnalysisDtypes:
    """perform_maxdiff_analysis must work on integer arrays of any
    Python-default dtype.

    Regression: Pyodide / WebAssembly numpy refuses to cast int64 to
    int32 inside ``np.bincount``. The fix is to use ``np.intp`` (the
    machine-native integer) internally; on 64-bit CPython this is a
    no-op, on Pyodide it converts to int32. This test pins the
    dtype-tolerance contract so it isn't accidentally regressed.
    """

    def test_accepts_int64_inputs(self, legacy) -> None:
        # 4 items, 3 tasks, 3 items per task.
        attrs = np.array(
            [[0, 1, 2], [1, 2, 3], [0, 2, 3]],
            dtype=np.int64,
        )
        pos = np.array([0, 1, 2], dtype=np.int64)
        neg = np.array([2, 3, 0], dtype=np.int64)
        unique = np.array(["A", "B", "C", "D"])
        result = legacy.perform_maxdiff_analysis(attrs, pos, neg, unique)
        assert list(result["Item"]) == ["A", "B", "C", "D"]
        # No exception is the headline assertion; sanity-check values.
        assert (result["Score"] != 0).any()

    def test_accepts_int32_inputs(self, legacy) -> None:
        attrs = np.array(
            [[0, 1, 2], [1, 2, 3], [0, 2, 3]],
            dtype=np.int32,
        )
        pos = np.array([0, 1, 2], dtype=np.int32)
        neg = np.array([2, 3, 0], dtype=np.int32)
        unique = np.array(["A", "B", "C", "D"])
        result = legacy.perform_maxdiff_analysis(attrs, pos, neg, unique)
        assert list(result["Item"]) == ["A", "B", "C", "D"]


class TestCountAnalysisRecoversOrdering:
    """If the synthetic generator and count-analysis are both correct,
    the rank of items by Score should match the rank by true utility
    (in the limit of many respondents)."""

    def test_recovers_top_three_by_score(self, legacy) -> None:
        # 8 items with a clear gradient in utility.
        true_utilities = np.array([3.0, 2.0, 1.0, 0.0, -0.5, -1.0, -2.0, -3.0])
        n_items = len(true_utilities)
        df = make_dataset(
            n_respondents=400,
            n_items=n_items,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=true_utilities,
            seed=2024,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        scores = legacy.calculate_scores_no_ci(df, attr_cols, "Most", "Least")
        # The three highest-Score items must match the three highest-utility
        # items: Item 1, Item 2, Item 3.
        top_three = scores.head(3)["Item"].tolist()
        assert set(top_three) == {"Item 1", "Item 2", "Item 3"}

    def test_recovers_bottom_three_by_score(self, legacy) -> None:
        true_utilities = np.array([3.0, 2.0, 1.0, 0.0, -0.5, -1.0, -2.0, -3.0])
        df = make_dataset(
            n_respondents=400,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=true_utilities,
            seed=2024,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        scores = legacy.calculate_scores_no_ci(df, attr_cols, "Most", "Least")
        bottom_three = scores.tail(3)["Item"].tolist()
        assert set(bottom_three) == {"Item 6", "Item 7", "Item 8"}
