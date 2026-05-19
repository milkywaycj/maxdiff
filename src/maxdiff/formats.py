"""Data format detection, conversion, and validation.

Two helpers (:func:`detect_terminology`, :func:`get_column_names`) plus
two classes (:class:`DataFormatDetector` with static utilities for
detecting and converting MaxDiff formats, and :func:`check_errors`
for input validation) live here. All depend only on pandas and
stdlib, so the module is safe for WebAssembly use.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd


def detect_terminology(df: pd.DataFrame) -> str | None:
    """Detect whether the data uses Most/Least or Best/Worst columns."""
    cols = [c.lower() for c in df.columns]
    if "most" in cols and "least" in cols:
        return "Most/Least"
    if "best" in cols and "worst" in cols:
        return "Best/Worst"
    return None


def get_column_names(df: pd.DataFrame, input_terminology: str) -> tuple[str, str]:
    """Return the actual case of the (positive, negative) columns."""
    cols_lower = {c.lower(): c for c in df.columns}
    if input_terminology == "Most/Least":
        return cols_lower.get("most", "Most"), cols_lower.get("least", "Least")
    return cols_lower.get("best", "Best"), cols_lower.get("worst", "Worst")


def check_errors(
    df: pd.DataFrame,
    attribute_columns: list[str],
    pos_col: str,
    neg_col: str,
) -> None:
    """Validate that a MaxDiff DataFrame is well-formed.

    Raises ``ValueError`` on any of:
    * inconsistent number of responses per participant
    * fewer than 3 attribute columns
    * missing data in attribute or selection columns
    * Most/Least choices not among the displayed attributes for that row

    Currently strict about balanced response counts; flexibility for
    partial completion is on the Phase 6 list.
    """
    responses_per_participant = df.groupby("Response ID").size()
    if not (responses_per_participant == responses_per_participant.iloc[0]).all():
        raise ValueError("Inconsistent number of responses per participant")
    if len(attribute_columns) < 3:
        raise ValueError("Less than 3 attributes found")
    columns_to_check = [*attribute_columns, pos_col, neg_col]
    if df[columns_to_check].isnull().any().any():
        raise ValueError(f"Missing data in {pos_col}, {neg_col}, or attribute columns")
    for idx, row in df.iterrows():
        displayed_attributes = set(row[attribute_columns])
        if row[pos_col] not in displayed_attributes:
            raise ValueError(f"Row {idx}: Selection '{row[pos_col]}' not in displayed attributes")
        if row[neg_col] not in displayed_attributes:
            raise ValueError(f"Row {idx}: Selection '{row[neg_col]}' not in displayed attributes")


class DataFormatDetector:
    """Detect and convert various MaxDiff data formats."""

    @staticmethod
    def detect_format(df: pd.DataFrame) -> tuple[str, str]:
        cols = [c.lower() for c in df.columns]
        col_list = list(df.columns)

        has_response_id = any("response" in c and "id" in c for c in cols) or "responseid" in cols
        has_attributes = any(c.startswith("attribute") for c in cols)
        has_most_least = ("most" in cols and "least" in cols) or (
            "best" in cols and "worst" in cols
        )

        if has_response_id and has_attributes and has_most_least:
            return "ready", "✓ Data appears to be in the correct format!"

        qualtrics_pattern = re.compile(r"(Q\d+|MD\d+|MaxDiff\d*)[-_]([\dA-Za-z]+)", re.IGNORECASE)
        qualtrics_matches = [c for c in col_list if qualtrics_pattern.match(c)]

        if len(qualtrics_matches) > 5:
            return (
                "qualtrics_wide",
                f"Detected Qualtrics-style format ({len(qualtrics_matches)} MaxDiff columns)",
            )

        long_indicators = ["item", "attribute", "option", "alternative", "choice"]
        selection_indicators = ["selected", "chosen", "response", "answer", "picked"]
        task_indicators = ["task", "set", "question", "trial", "screen"]

        has_item_col = any(any(ind in c for ind in long_indicators) for c in cols)
        has_selection_col = any(any(ind in c for ind in selection_indicators) for c in cols)
        has_task_col = any(any(ind in c for ind in task_indicators) for c in cols)

        if has_item_col and (has_selection_col or has_task_col):
            return "long", "Detected long format (one row per item)"

        task_pattern = re.compile(
            r"(task|set|q)[-_]?(\d+)[-_]?(attr|item|opt|best|worst|most|least|\d+)",
            re.IGNORECASE,
        )
        task_matches = [c for c in col_list if task_pattern.match(c)]

        if len(task_matches) > 5:
            return "wide_by_respondent", f"Detected wide format ({len(task_matches)} task columns)"

        return "unknown", "Could not auto-detect format. Please use manual column mapping."

    @staticmethod
    def get_format_description(format_type: str) -> str:
        descriptions = {
            "ready": "Your data is already in the correct format! You can proceed directly to analysis.",
            "qualtrics_wide": "Qualtrics format detected. The converter will reshape columns like Q1_1, Q1_2, Q1_Best into the required format.",
            "long": "Long format detected (one row per item). The converter will pivot this into wide format.",
            "wide_by_respondent": "Wide-by-respondent format detected. The converter will reshape into multiple rows per respondent.",
            "unknown": "Format not recognized. Use the Column Mapper to manually specify your columns.",
        }
        return descriptions.get(format_type, "")

    @staticmethod
    def convert_qualtrics_wide(
        df: pd.DataFrame,
        task_prefix: str = "Q",
        best_suffix: str = "Best",
        worst_suffix: str = "Worst",
        id_column: str | None = None,
    ) -> pd.DataFrame:
        if id_column is None:
            id_candidates = [c for c in df.columns if "response" in c.lower() or "id" in c.lower()]
            id_column = id_candidates[0] if id_candidates else df.columns[0]

        task_pattern = re.compile(
            rf"({task_prefix}\d+)[-_](\d+|{best_suffix}|{worst_suffix}|Best|Worst|Most|Least)",
            re.IGNORECASE,
        )

        tasks: dict[str, dict[str, str]] = defaultdict(dict)
        for col in df.columns:
            match = task_pattern.match(col)
            if match:
                task_name = match.group(1).upper()
                suffix = match.group(2)
                tasks[task_name][suffix] = col

        if not tasks:
            raise ValueError(f"No task columns found matching pattern '{task_prefix}N_X'")

        rows = []
        for _, respondent in df.iterrows():
            resp_id = respondent[id_column]

            for task_name in sorted(tasks.keys()):
                task_cols = tasks[task_name]

                attr_cols = {
                    k: v
                    for k, v in task_cols.items()
                    if k.isdigit()
                    or k.lower()
                    not in [
                        "best",
                        "worst",
                        "most",
                        "least",
                        best_suffix.lower(),
                        worst_suffix.lower(),
                    ]
                }

                best_col = worst_col = None
                for suffix, col in task_cols.items():
                    if suffix.lower() in ["best", "most", best_suffix.lower()]:
                        best_col = col
                    elif suffix.lower() in ["worst", "least", worst_suffix.lower()]:
                        worst_col = col

                if not attr_cols or not best_col or not worst_col:
                    continue

                row = {"Response ID": resp_id}
                for i, (_suffix, col) in enumerate(sorted(attr_cols.items()), 1):
                    row[f"Attribute{i}"] = respondent[col]

                row["Most"] = respondent[best_col]
                row["Least"] = respondent[worst_col]
                rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def convert_long_format(
        df: pd.DataFrame,
        id_col: str,
        task_col: str,
        item_col: str,
        selection_col: str,
        most_value: str = "Most",
        least_value: str = "Least",
    ) -> pd.DataFrame:
        rows = []
        grouped = df.groupby([id_col, task_col])

        for (resp_id, _task), group in grouped:
            row = {"Response ID": resp_id}

            items = group[item_col].tolist()
            for i, item in enumerate(items, 1):
                row[f"Attribute{i}"] = item

            most_mask = (
                group[selection_col]
                .astype(str)
                .str.lower()
                .str.contains(most_value.lower(), na=False)
            )
            least_mask = (
                group[selection_col]
                .astype(str)
                .str.lower()
                .str.contains(least_value.lower(), na=False)
            )

            most_item = group.loc[most_mask, item_col]
            least_item = group.loc[least_mask, item_col]

            row["Most"] = most_item.iloc[0] if len(most_item) > 0 else None
            row["Least"] = least_item.iloc[0] if len(least_item) > 0 else None

            rows.append(row)

        return pd.DataFrame(rows)
