"""Display statistics and balance diagnostics.

Two functions:

  * :func:`calculate_display_statistics` - per-item display counts,
    best / worst rates, and overall balance metrics.

  * :func:`format_display_report` - format the stats as a human-readable
    text report. Used by both the desktop GUI's log pane and the
    browser tool's log.
"""

from __future__ import annotations

import pandas as pd


def calculate_display_statistics(
    df: pd.DataFrame,
    attribute_columns: list[str],
    pos_col: str,
    neg_col: str,
) -> tuple[pd.DataFrame, dict]:
    """Compute display counts, selection counts, and balance metrics.

    Returns
    -------
    (pd.DataFrame, dict)
        First element: per-item statistics with columns
        ``Item, Times Displayed, Times Selected Best, Times Selected
        Worst, Times Unselected, Best Rate, Worst Rate``. Sorted by
        Times Displayed descending.
        Second element: aggregate balance metrics including counts,
        coefficient of variation, and warnings about under-displayed
        or never-selected items.
    """
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]

    stats = []
    for item in unique_items:
        displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()

        stats.append(
            {
                "Item": item,
                "Times Displayed": displays,
                "Times Selected Best": pos_count,
                "Times Selected Worst": neg_count,
                "Times Unselected": displays - pos_count - neg_count,
                "Best Rate": pos_count / displays if displays > 0 else 0,
                "Worst Rate": neg_count / displays if displays > 0 else 0,
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("Times Displayed", ascending=False)

    display_counts = stats_df["Times Displayed"].values
    mean_displays = display_counts.mean() if len(display_counts) > 0 else 0
    std_displays = display_counts.std() if len(display_counts) > 0 else 0

    balance_metrics = {
        "total_displays": int(display_counts.sum()),
        "num_items": len(unique_items),
        "min_displays": int(display_counts.min()) if len(display_counts) > 0 else 0,
        "max_displays": int(display_counts.max()) if len(display_counts) > 0 else 0,
        "mean_displays": float(mean_displays),
        "std_displays": float(std_displays),
        "cv_displays": float(std_displays / mean_displays) if mean_displays > 0 else 0,
        "range_displays": int(display_counts.max() - display_counts.min())
        if len(display_counts) > 0
        else 0,
        "is_balanced": False,
        "balance_warnings": [],
        "balance_status": "Unknown",
    }

    warnings: list[str] = []
    cv = balance_metrics["cv_displays"]
    if cv < 0.01:
        balance_metrics["balance_status"] = "Perfectly Balanced"
        balance_metrics["is_balanced"] = True
    elif cv < 0.05:
        balance_metrics["balance_status"] = "Well Balanced"
        balance_metrics["is_balanced"] = True
    elif cv < 0.10:
        balance_metrics["balance_status"] = "Reasonably Balanced"
        balance_metrics["is_balanced"] = True
        warnings.append(f"Minor imbalance detected (CV={cv:.1%}). Results are still valid.")
    elif cv < 0.20:
        balance_metrics["balance_status"] = "Somewhat Unbalanced"
        balance_metrics["is_balanced"] = False
        warnings.append(
            f"⚠️ Moderate imbalance detected (CV={cv:.1%}). Consider this when interpreting results."
        )
    else:
        balance_metrics["balance_status"] = "Highly Unbalanced"
        balance_metrics["is_balanced"] = False
        warnings.append(f"⚠️ HIGH IMBALANCE detected (CV={cv:.1%}). Results may be biased!")

    mean_disp = balance_metrics["mean_displays"]
    std_disp = balance_metrics["std_displays"]

    if std_disp > 0:
        under_displayed = stats_df[stats_df["Times Displayed"] < mean_disp - 2 * std_disp][
            "Item"
        ].tolist()
        over_displayed = stats_df[stats_df["Times Displayed"] > mean_disp + 2 * std_disp][
            "Item"
        ].tolist()

        if under_displayed:
            items_str = ", ".join(str(x) for x in under_displayed[:5])
            warnings.append(f"⚠️ Under-displayed items (>2 SD below mean): {items_str}")
            if len(under_displayed) > 5:
                warnings[-1] += f" and {len(under_displayed) - 5} more"

        if over_displayed:
            items_str = ", ".join(str(x) for x in over_displayed[:5])
            warnings.append(f"⚠️ Over-displayed items (>2 SD above mean): {items_str}")
            if len(over_displayed) > 5:
                warnings[-1] += f" and {len(over_displayed) - 5} more"

    low_threshold = 30
    low_display_items = stats_df[stats_df["Times Displayed"] < low_threshold]["Item"].tolist()
    if low_display_items:
        items_str = ", ".join(str(x) for x in low_display_items[:5])
        warnings.append(f"⚠️ Low display counts (<{low_threshold}): {items_str}")
        if len(low_display_items) > 5:
            warnings[-1] += f" and {len(low_display_items) - 5} more"
        warnings.append("   Low counts may lead to unreliable estimates for these items.")

    never_best = stats_df[stats_df["Times Selected Best"] == 0]["Item"].tolist()
    never_worst = stats_df[stats_df["Times Selected Worst"] == 0]["Item"].tolist()

    if never_best:
        items_str = ", ".join(str(x) for x in never_best[:5])
        warnings.append(f"ℹ️ Items never selected as best: {items_str}")
        if len(never_best) > 5:
            warnings[-1] += f" and {len(never_best) - 5} more"

    if never_worst:
        items_str = ", ".join(str(x) for x in never_worst[:5])
        warnings.append(f"ℹ️ Items never selected as worst: {items_str}")
        if len(never_worst) > 5:
            warnings[-1] += f" and {len(never_worst) - 5} more"

    balance_metrics["balance_warnings"] = warnings

    return stats_df, balance_metrics


def format_display_report(
    stats_df: pd.DataFrame, balance_metrics: dict, output_terms: tuple[str, str]
) -> str:
    """Format display statistics as a multi-line text report.

    Used by both the desktop GUI's log pane and the browser tool's
    log; pure text output, no GUI dependency.
    """
    pos_label, neg_label = output_terms

    lines = []
    lines.append("=" * 60)
    lines.append("📊 DISPLAY STATISTICS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total items: {balance_metrics['num_items']}")
    lines.append(f"Total displays across all tasks: {balance_metrics['total_displays']:,}")
    lines.append("")
    lines.append("BALANCE ASSESSMENT:")
    lines.append(f"  Status: {balance_metrics['balance_status']}")
    lines.append(f"  Min displays: {balance_metrics['min_displays']:,}")
    lines.append(f"  Max displays: {balance_metrics['max_displays']:,}")
    lines.append(f"  Mean displays: {balance_metrics['mean_displays']:,.1f}")
    lines.append(f"  Std deviation: {balance_metrics['std_displays']:,.1f}")
    lines.append(f"  Coefficient of variation: {balance_metrics['cv_displays']:.1%}")
    lines.append("")

    if balance_metrics["balance_warnings"]:
        lines.append("WARNINGS & NOTES:")
        for warning in balance_metrics["balance_warnings"]:
            lines.append(f"  {warning}")
        lines.append("")

    lines.append("DISPLAY COUNTS PER ITEM:")
    lines.append("-" * 60)
    lines.append(f"{'Item':<30} {'Displays':>10} {pos_label:>10} {neg_label:>10}")
    lines.append("-" * 60)

    for _, row in stats_df.iterrows():
        item_name = str(row["Item"])[:28]
        lines.append(
            f"{item_name:<30} {row['Times Displayed']:>10,} "
            f"{row['Times Selected Best']:>10,} {row['Times Selected Worst']:>10,}"
        )

    lines.append("-" * 60)
    lines.append("")

    return "\n".join(lines)
