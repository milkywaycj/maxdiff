"""Bootstrap confidence intervals for count-based MaxDiff scores.

:func:`bootstrap_analysis` resamples respondents with replacement,
recomputes the per-item net score on each resample, and reports the
2.5 / 97.5 percentiles of the resulting distribution alongside the
observed score on the original sample.

Resampling is at the respondent level (not the row level) to preserve
within-respondent task clustering.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from maxdiff.count import perform_maxdiff_analysis


def bootstrap_analysis(
    df: pd.DataFrame,
    attribute_columns: list[str],
    unique_attributes: np.ndarray,
    attr_to_index: dict,
    pos_col: str,
    neg_col: str,
    n_iterations: int = 10000,
    progress_callback: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Run bootstrap resampling for the MaxDiff count analysis.

    Parameters
    ----------
    df
        Long-format MaxDiff data with ``Response ID``.
    attribute_columns
        Names of columns holding items shown per task.
    unique_attributes
        1-D array of all distinct items, in a fixed order.
    attr_to_index
        Mapping from item label to its integer index in
        ``unique_attributes``.
    pos_col, neg_col
        Column names for best / worst choices.
    n_iterations
        Number of bootstrap resamples.
    progress_callback
        Optional callable accepting a float in ``[0, 1]``. Called
        every 100 iterations and once at the end.

    Returns
    -------
    pd.DataFrame
        Columns: ``Item, Score, 2.5th Percentile, 97.5th Percentile,
        Negative Error, Positive Error``. Sorted by ``Score``
        descending. ``Score`` is the score on the original sample
        (not the bootstrap mean).
    """
    unique_participants = df["Response ID"].unique()
    n_participants = len(unique_participants)
    n_attributes = len(unique_attributes)
    df_reset = df.reset_index(drop=True)
    attribute_data = df_reset[attribute_columns].replace(attr_to_index).values
    pos_data = df_reset[pos_col].replace(attr_to_index).values
    neg_data = df_reset[neg_col].replace(attr_to_index).values
    participant_indices = df_reset.groupby("Response ID").indices
    observed_results = perform_maxdiff_analysis(
        attribute_data, pos_data, neg_data, unique_attributes
    )
    observed_scores = observed_results.set_index("Item")["Score"]
    all_scores = np.zeros((n_iterations, n_attributes))
    rng = np.random.default_rng()

    for i in range(n_iterations):
        sampled_participants = rng.choice(unique_participants, size=n_participants, replace=True)
        sampled_indices = np.concatenate([participant_indices[p] for p in sampled_participants])
        results = perform_maxdiff_analysis(
            attribute_data[sampled_indices],
            pos_data[sampled_indices],
            neg_data[sampled_indices],
            unique_attributes,
        )
        all_scores[i] = results["Score"].values
        if progress_callback and i % 100 == 0:
            progress_callback(i / n_iterations)

    if progress_callback:
        progress_callback(1.0)

    percentile_2_5 = np.percentile(all_scores, 2.5, axis=0)
    percentile_97_5 = np.percentile(all_scores, 97.5, axis=0)
    obs_scores_array = observed_scores[unique_attributes].values

    return pd.DataFrame(
        {
            "Item": unique_attributes,
            "Score": obs_scores_array,
            "2.5th Percentile": percentile_2_5,
            "97.5th Percentile": percentile_97_5,
            "Negative Error": obs_scores_array - percentile_2_5,
            "Positive Error": percentile_97_5 - obs_scores_array,
        }
    ).sort_values("Score", ascending=False)
