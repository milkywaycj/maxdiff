"""Characterization tests for ``DataFormatDetector``.

The detector classifies an incoming DataFrame into one of:

  * "ready"            - already has Response ID + Attribute* + Most/Best + Least/Worst
  * "qualtrics_wide"   - Q1_1, Q1_2, ..., Q1_Best, Q1_Worst style
  * "long"             - one row per item, with a "selected" indicator column
  * "wide_by_respondent" - task-prefixed columns with per-task attribute slots
  * "unknown"          - no pattern matched

It also exposes two converters:

  * convert_qualtrics_wide
  * convert_long_format

Tests cover the detector's classifications on representative inputs
and the converters' shape and content correctness on the formats they
claim to handle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import legacy_loader


@pytest.fixture(scope="module")
def legacy():
    return legacy_loader.load_analyzer()


# ----------------------------------------------------------------------
# detect_format
# ----------------------------------------------------------------------


class TestDetectFormatReady:
    def test_recognizes_canonical_most_least(self, legacy) -> None:
        df = pd.DataFrame(
            columns=["Response ID", "Attribute1", "Attribute2", "Attribute3", "Most", "Least"]
        )
        fmt, _msg = legacy.DataFormatDetector.detect_format(df)
        assert fmt == "ready"

    def test_recognizes_canonical_best_worst(self, legacy) -> None:
        df = pd.DataFrame(
            columns=["Response ID", "Attribute1", "Attribute2", "Attribute3", "Best", "Worst"]
        )
        fmt, _msg = legacy.DataFormatDetector.detect_format(df)
        assert fmt == "ready"


class TestDetectFormatQualtrics:
    def test_recognizes_qualtrics_wide_columns(self, legacy) -> None:
        cols = ["Response ID"]
        for q in range(1, 7):
            for i in range(1, 5):
                cols.append(f"Q{q}_{i}")
            cols.extend([f"Q{q}_Best", f"Q{q}_Worst"])
        df = pd.DataFrame(columns=cols)
        fmt, _msg = legacy.DataFormatDetector.detect_format(df)
        assert fmt == "qualtrics_wide"


class TestDetectFormatUnknown:
    def test_random_columns_are_unknown(self, legacy) -> None:
        df = pd.DataFrame(columns=["foo", "bar", "baz"])
        fmt, _msg = legacy.DataFormatDetector.detect_format(df)
        assert fmt == "unknown"


class TestGetFormatDescription:
    @pytest.mark.parametrize(
        "fmt",
        ["ready", "qualtrics_wide", "long", "wide_by_respondent", "unknown"],
    )
    def test_returns_non_empty_description_for_each_format(self, legacy, fmt) -> None:
        text = legacy.DataFormatDetector.get_format_description(fmt)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_returns_empty_for_unrecognized_format(self, legacy) -> None:
        text = legacy.DataFormatDetector.get_format_description("nonsense")
        assert text == ""


# ----------------------------------------------------------------------
# convert_qualtrics_wide
# ----------------------------------------------------------------------


class TestConvertQualtricsWide:
    def _qualtrics_df(self) -> pd.DataFrame:
        """Build a small two-respondent, two-task Qualtrics-style frame."""
        return pd.DataFrame(
            [
                {
                    "Response ID": "R1",
                    "Q1_1": "A",
                    "Q1_2": "B",
                    "Q1_3": "C",
                    "Q1_Best": "A",
                    "Q1_Worst": "C",
                    "Q2_1": "B",
                    "Q2_2": "C",
                    "Q2_3": "D",
                    "Q2_Best": "B",
                    "Q2_Worst": "D",
                },
                {
                    "Response ID": "R2",
                    "Q1_1": "A",
                    "Q1_2": "B",
                    "Q1_3": "C",
                    "Q1_Best": "C",
                    "Q1_Worst": "A",
                    "Q2_1": "B",
                    "Q2_2": "C",
                    "Q2_3": "D",
                    "Q2_Best": "D",
                    "Q2_Worst": "C",
                },
            ]
        )

    def test_returns_dataframe(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_qualtrics_wide(self._qualtrics_df())
        assert isinstance(result, pd.DataFrame)

    def test_output_has_expected_columns(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_qualtrics_wide(self._qualtrics_df())
        for col in ["Response ID", "Attribute1", "Attribute2", "Attribute3", "Most", "Least"]:
            assert col in result.columns

    def test_row_count_is_respondents_times_tasks(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_qualtrics_wide(self._qualtrics_df())
        # 2 respondents x 2 tasks = 4 rows
        assert len(result) == 4

    def test_most_and_least_carried_over_correctly(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_qualtrics_wide(self._qualtrics_df())
        # R1, Q1: Best=A, Worst=C
        r1_q1 = result[(result["Response ID"] == "R1")].iloc[0]
        assert r1_q1["Most"] == "A"
        assert r1_q1["Least"] == "C"

    def test_raises_when_no_task_columns_match(self, legacy) -> None:
        df = pd.DataFrame([{"Response ID": "R1", "Foo": "X", "Bar": "Y"}])
        with pytest.raises(ValueError, match="No task columns"):
            legacy.DataFormatDetector.convert_qualtrics_wide(df)


# ----------------------------------------------------------------------
# convert_long_format
# ----------------------------------------------------------------------


class TestConvertLongFormat:
    def _long_df(self) -> pd.DataFrame:
        rows = []
        # Two respondents, two tasks each, three items per task.
        for resp in ["R1", "R2"]:
            for task in [1, 2]:
                items = ["A", "B", "C"] if task == 1 else ["B", "C", "D"]
                most = items[0]
                least = items[-1]
                for item in items:
                    selection = "Most" if item == most else ("Least" if item == least else "")
                    rows.append(
                        {
                            "ID": resp,
                            "Task": task,
                            "Item": item,
                            "Selection": selection,
                        }
                    )
        return pd.DataFrame(rows)

    def test_pivots_into_one_row_per_task(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_long_format(
            self._long_df(), "ID", "Task", "Item", "Selection"
        )
        # 2 respondents x 2 tasks = 4 rows
        assert len(result) == 4
        for col in ["Response ID", "Attribute1", "Attribute2", "Attribute3", "Most", "Least"]:
            assert col in result.columns

    def test_most_and_least_extracted(self, legacy) -> None:
        result = legacy.DataFormatDetector.convert_long_format(
            self._long_df(), "ID", "Task", "Item", "Selection"
        )
        # In the constructed long frame, for each (resp, task) the first
        # item was marked Most and the last was Least. Each (resp, task)
        # row in the output must reflect that.
        r1_t1 = result[result["Response ID"] == "R1"].iloc[0]
        assert r1_t1["Most"] == "A"
        assert r1_t1["Least"] == "C"
