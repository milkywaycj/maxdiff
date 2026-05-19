"""Characterization tests for the small helper functions:

  * process_color_input
  * detect_terminology
  * get_column_names
  * check_errors
  * calculate_correlation_matrix

These functions are short and have well-defined contracts. Pinning
their behavior here covers a non-trivial fraction of the analyzer's
line count and protects the Phase 3 extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader


@pytest.fixture(scope="module")
def legacy():
    return legacy_loader.load_analyzer()


# ----------------------------------------------------------------------
# process_color_input
# ----------------------------------------------------------------------


class TestProcessColorInput:
    @pytest.mark.parametrize(
        ("user_input", "expected"),
        [
            ("#FF0000", "#FF0000"),
            ("#abcdef", "#abcdef"),
            ("FF0000", "#FF0000"),  # 6-char hex without leading hash gets one
            ("aabbcc", "#aabbcc"),
            ("red", "red"),  # CSS4 named color
            ("blue", "blue"),
            ("black", "black"),
        ],
    )
    def test_accepts_valid_color_inputs(self, legacy, user_input: str, expected: str) -> None:
        result = legacy.process_color_input(user_input)
        assert result == expected

    def test_returns_none_for_unrecognized_color_name(self, legacy) -> None:
        assert legacy.process_color_input("not_a_color") is None
        assert legacy.process_color_input("12345") is None  # 5 chars, neither hex nor name

    def test_returns_hash_prefix_input_unchanged_even_if_invalid_hex(self, legacy) -> None:
        """Pin the CURRENT (buggy) behavior of process_color_input.

        The function returns any string starting with ``#`` unchanged,
        without validating the remaining characters are valid hex. The
        ``test_should_reject_invalid_hex_after_hash`` test below
        documents the corrected behavior we want. Phase 6 fixes the
        bug; when it does, the xfail flips to xpass and (per the
        suite's xfail_strict setting) CI will fail, prompting both
        tests to be updated together.
        """
        assert legacy.process_color_input("#GGGGGG") == "#GGGGGG"
        assert legacy.process_color_input("#zzz") == "#zzz"

    @pytest.mark.xfail(
        reason="Bug: process_color_input accepts any string starting with '#' without "
        "validating hex characters. Fix scheduled for Phase 6.",
        strict=True,
    )
    def test_should_reject_invalid_hex_after_hash(self, legacy) -> None:
        assert legacy.process_color_input("#GGGGGG") is None
        assert legacy.process_color_input("#zzz") is None

    def test_returns_none_for_empty_input(self, legacy) -> None:
        assert legacy.process_color_input("") is None
        assert legacy.process_color_input(None) is None

    def test_handles_whitespace(self, legacy) -> None:
        # Leading/trailing whitespace should be stripped.
        assert legacy.process_color_input("  red  ") == "red"


# ----------------------------------------------------------------------
# detect_terminology + get_column_names
# ----------------------------------------------------------------------


class TestTerminologyDetection:
    def test_detects_most_least_terminology(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "Most", "Least"])
        assert legacy.detect_terminology(df) == "Most/Least"

    def test_detects_best_worst_terminology(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "Best", "Worst"])
        assert legacy.detect_terminology(df) == "Best/Worst"

    def test_returns_none_for_unknown_columns(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "Chosen", "Rejected"])
        assert legacy.detect_terminology(df) is None

    def test_is_case_insensitive(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "BEST", "worst"])
        assert legacy.detect_terminology(df) == "Best/Worst"


class TestGetColumnNames:
    def test_returns_actual_casing_of_most_least(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "Most", "Least"])
        pos, neg = legacy.get_column_names(df, "Most/Least")
        assert pos == "Most"
        assert neg == "Least"

    def test_returns_actual_casing_when_columns_differ_in_case(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "MOST", "least"])
        pos, neg = legacy.get_column_names(df, "Most/Least")
        assert pos == "MOST"
        assert neg == "least"

    def test_returns_best_worst_with_correct_casing(self, legacy) -> None:
        df = pd.DataFrame(columns=["Response ID", "Attribute1", "Best", "Worst"])
        pos, neg = legacy.get_column_names(df, "Best/Worst")
        assert pos == "Best"
        assert neg == "Worst"


# ----------------------------------------------------------------------
# check_errors
# ----------------------------------------------------------------------


class TestCheckErrors:
    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
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
                    "Attribute1": "B",
                    "Attribute2": "C",
                    "Attribute3": "D",
                    "Most": "C",
                    "Least": "D",
                },
                {
                    "Response ID": "R2",
                    "Attribute1": "A",
                    "Attribute2": "B",
                    "Attribute3": "D",
                    "Most": "A",
                    "Least": "D",
                },
                {
                    "Response ID": "R2",
                    "Attribute1": "B",
                    "Attribute2": "C",
                    "Attribute3": "D",
                    "Most": "C",
                    "Least": "B",
                },
            ]
        )

    def test_accepts_valid_data(self, legacy) -> None:
        df = self._valid_df()
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        # Should not raise.
        legacy.check_errors(df, attr_cols, "Most", "Least")

    def test_rejects_inconsistent_responses_per_participant(self, legacy) -> None:
        df = pd.concat(
            [
                self._valid_df(),
                pd.DataFrame(
                    [
                        {
                            "Response ID": "R3",
                            "Attribute1": "A",
                            "Attribute2": "B",
                            "Attribute3": "C",
                            "Most": "A",
                            "Least": "C",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        attr_cols = ["Attribute1", "Attribute2", "Attribute3"]
        with pytest.raises(ValueError, match="Inconsistent"):
            legacy.check_errors(df, attr_cols, "Most", "Least")

    def test_rejects_fewer_than_three_attributes(self, legacy) -> None:
        df = pd.DataFrame(
            [
                {
                    "Response ID": "R1",
                    "Attribute1": "A",
                    "Attribute2": "B",
                    "Most": "A",
                    "Least": "B",
                },
                {
                    "Response ID": "R1",
                    "Attribute1": "B",
                    "Attribute2": "C",
                    "Most": "B",
                    "Least": "C",
                },
            ]
        )
        with pytest.raises(ValueError, match="Less than 3"):
            legacy.check_errors(df, ["Attribute1", "Attribute2"], "Most", "Least")

    def test_rejects_missing_data(self, legacy) -> None:
        df = self._valid_df()
        df.loc[0, "Most"] = np.nan
        with pytest.raises(ValueError, match="Missing data"):
            legacy.check_errors(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")

    def test_rejects_most_not_in_displayed_attributes(self, legacy) -> None:
        df = self._valid_df()
        df.loc[0, "Most"] = "Z"  # Not in any Attribute column
        with pytest.raises(ValueError, match="not in displayed attributes"):
            legacy.check_errors(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")

    def test_rejects_least_not_in_displayed_attributes(self, legacy) -> None:
        df = self._valid_df()
        df.loc[0, "Least"] = "Z"
        with pytest.raises(ValueError, match="not in displayed attributes"):
            legacy.check_errors(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")


# ----------------------------------------------------------------------
# calculate_correlation_matrix
# ----------------------------------------------------------------------


class TestCorrelationMatrix:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
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
                    "Least": "D",
                },
                {
                    "Response ID": "R2",
                    "Attribute1": "A",
                    "Attribute2": "B",
                    "Attribute3": "C",
                    "Most": "B",
                    "Least": "A",
                },
                {
                    "Response ID": "R2",
                    "Attribute1": "A",
                    "Attribute2": "C",
                    "Attribute3": "D",
                    "Most": "C",
                    "Least": "A",
                },
            ]
        )

    def test_returns_square_matrix(self, legacy) -> None:
        df = self._df()
        corr = legacy.calculate_correlation_matrix(
            df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least"
        )
        # Square with one row/column per unique item.
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] >= 4  # A, B, C, D at minimum

    def test_diagonal_is_one(self, legacy) -> None:
        df = self._df()
        corr = legacy.calculate_correlation_matrix(
            df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least"
        )
        diag = np.diag(corr.values)
        # An item is perfectly correlated with itself.
        np.testing.assert_allclose(diag, 1.0)

    def test_matrix_is_symmetric(self, legacy) -> None:
        df = self._df()
        corr = legacy.calculate_correlation_matrix(
            df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least"
        )
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_values_lie_in_minus_one_to_one(self, legacy) -> None:
        df = self._df()
        corr = legacy.calculate_correlation_matrix(
            df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least"
        )
        # Pearson correlations from -1 to +1 (modulo NaN for constant series).
        finite = corr.values[np.isfinite(corr.values)]
        assert np.all(finite >= -1.0 - 1e-9)
        assert np.all(finite <= 1.0 + 1e-9)
