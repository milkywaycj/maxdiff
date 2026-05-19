"""Count-based MaxDiff analysis.

Three functions implement the classic count-based scoring:

  * :func:`calculate_observed_percentages` - produces percent-best,
    percent-worst, percent-unselected, and the net score (% best
    minus % worst times 100) per item, sorted by score.

  * :func:`calculate_scores_no_ci` - just the net score per item,
    sorted; this is the same score as in the bootstrap output and
    in :func:`calculate_observed_percentages`.

  * :func:`perform_maxdiff_analysis` - the numpy-array core used
    inside :func:`bootstrap_analysis` for speed. Operates on
    pre-encoded integer arrays rather than the original DataFrame.

These functions form the simplest, most-transparent rung of the
analysis pipeline. Research consistently shows count-based scores
correlate r > 0.99 with Hierarchical Bayes utilities on standard
balanced MaxDiff designs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_observed_percentages(
    df: pd.DataFrame,
    attribute_columns: list[str],
    pos_col: str,
    neg_col: str,
    output_terms: tuple[str, str],
) -> pd.DataFrame:
    """Compute per-item best/worst/unselected percentages and net score.

    Parameters
    ----------
    df
        Long-format MaxDiff data with ``Response ID``, attribute
        columns, and the ``pos_col`` / ``neg_col`` choice columns.
    attribute_columns
        Names of the columns containing the items shown in each task.
    pos_col, neg_col
        Column names holding the best / worst (or most / least) choice.
    output_terms
        Tuple ``(pos_label, neg_label)`` used to name the output
        columns; e.g. ``("Best", "Worst")``.

    Returns
    -------
    pd.DataFrame
        One row per item, sorted by ``Score`` descending.
    """
    pos_label, neg_label = output_terms
    results = []
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]

    for item in unique_items:
        item_displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()
        unselected_count = item_displays - pos_count - neg_count
        results.append(
            {
                "Item": item,
                f"% Selected as {pos_label}": pos_count / item_displays * 100
                if item_displays > 0
                else 0,
                "% Unselected": unselected_count / item_displays * 100 if item_displays > 0 else 0,
                f"% Selected as {neg_label}": neg_count / item_displays * 100
                if item_displays > 0
                else 0,
                "Score": (pos_count - neg_count) / item_displays * 100 if item_displays > 0 else 0,
            }
        )
    return pd.DataFrame(results).sort_values("Score", ascending=False)


def calculate_scores_no_ci(
    df: pd.DataFrame,
    attribute_columns: list[str],
    pos_col: str,
    neg_col: str,
) -> pd.DataFrame:
    """Compute the net (best minus worst) score per item.

    Returns a DataFrame with columns ``Item`` and ``Score``, sorted by
    ``Score`` descending. The same score is produced by
    :func:`calculate_observed_percentages` but this function skips the
    breakdown columns for speed.
    """
    unique_attributes = pd.unique(df[attribute_columns].values.ravel())
    unique_attributes = [attr for attr in unique_attributes if pd.notna(attr)]
    results = []
    for item in unique_attributes:
        item_displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()
        results.append(
            {
                "Item": item,
                "Score": (pos_count - neg_count) / item_displays * 100 if item_displays > 0 else 0,
            }
        )
    return pd.DataFrame(results).sort_values("Score", ascending=False)


def perform_maxdiff_analysis(
    attribute_data: np.ndarray,
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    unique_attributes: np.ndarray,
) -> pd.DataFrame:
    """Vectorized count-analysis on pre-encoded integer arrays.

    Used by :func:`bootstrap_analysis`. ``attribute_data`` is a
    ``(n_tasks, n_items_per_task)`` array of item indices,
    ``pos_data`` / ``neg_data`` are length-``n_tasks`` arrays of
    item indices for the best / worst choices.

    The bincount inputs are cast to ``np.intp`` (machine-native
    integer) before counting. On 64-bit CPython this is int64 and
    a no-op; on Pyodide / WebAssembly it's int32, and the cast is
    required because Pyodide's numpy refuses to silently downcast
    int64 -> int32 in bincount.
    """
    n_attributes = len(unique_attributes)
    display_count = np.sum(attribute_data[:, :, None] == np.arange(n_attributes), axis=(0, 1))
    pos_count = np.bincount(np.asarray(pos_data, dtype=np.intp), minlength=n_attributes)
    neg_count = np.bincount(np.asarray(neg_data, dtype=np.intp), minlength=n_attributes)

    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(
            display_count > 0,
            ((pos_count / display_count) - (neg_count / display_count)) * 100,
            0,
        )

    return pd.DataFrame({"Item": unique_attributes, "Score": score})
