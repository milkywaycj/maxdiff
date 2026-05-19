"""Item-level correlation matrix from best/worst selection patterns.

For each respondent, accumulate +1 for every "best" pick and -1 for
every "worst" pick on each item, then compute the Pearson
correlation across respondents. Items chosen together as best (or
together as worst) by the same respondents end up positively
correlated; items that one respondent ranks high and another ranks
low end up near zero or negative.

Known caveat: items that are never displayed to a given respondent
contribute a 0 (treated as "neutral") rather than a missing value,
which slightly understates correlations. Fix scheduled for Phase 6.
"""

from __future__ import annotations

import pandas as pd


def calculate_correlation_matrix(
    df: pd.DataFrame,
    attribute_columns: list[str],
    pos_col: str,
    neg_col: str,
) -> pd.DataFrame:
    """Compute item-by-item correlation matrix from best/worst patterns.

    Returns a square DataFrame indexed and labelled by item, with
    Pearson correlation coefficients in each cell. Diagonal is 1.0;
    matrix is symmetric.
    """
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]
    count_df = pd.DataFrame(index=df["Response ID"].unique(), columns=unique_items, data=0)

    for _, row in df.iterrows():
        count_df.loc[row["Response ID"], row[pos_col]] += 1
        count_df.loc[row["Response ID"], row[neg_col]] -= 1

    return count_df.astype(float).corr()
