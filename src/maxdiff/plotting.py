"""Thread-safe matplotlib plotting helpers.

The desktop GUI runs analysis (including plotting) from a worker
thread so the UI stays responsive. Matplotlib's ``pyplot`` module
keeps thread-unsafe global state (current figure registry, current
axes, font cache locks) and can corrupt or crash when used from a
non-main thread. The Phase 6 rewrite of this module uses the OO API
exclusively: figures are built with :class:`matplotlib.figure.Figure`
attached to a :class:`FigureCanvasAgg`, axes are created with
``fig.add_subplot``, and figures are saved via ``fig.savefig`` and
discarded by dropping the reference.

This module also drops the legacy seaborn dependency. The
correlation heatmap is rendered with ``ax.imshow`` + a colorbar
attached to the axes, producing visually equivalent output without
pulling in seaborn.

All plot functions return the :class:`matplotlib.figure.Figure`
instance, which the caller can save with :func:`save_plot` or
inspect directly. Callers should not need to call ``plt.close``;
the figure is held only by the returned reference.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _new_figure(*, figsize: tuple[float, float], dpi: int = 100) -> Figure:
    """Create a thread-safe Figure (no pyplot involvement)."""
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)  # attach a canvas; required for savefig in some backends
    return fig


# ----------------------------------------------------------------------
# Public plotting API
# ----------------------------------------------------------------------


def plot_display_balance(
    stats_df: pd.DataFrame,
    title: str = "Item Display Frequency",
    output_terms: tuple[str, str] = ("Best", "Worst"),
) -> Figure:
    """Bar chart of per-item display counts and selection rates."""
    pos_label, neg_label = output_terms

    stats_sorted = stats_df.sort_values("Times Displayed", ascending=True)
    fig = _new_figure(figsize=(14, max(6, len(stats_df) * 0.3)))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    mean_val = stats_df["Times Displayed"].mean()
    colors = ["#4CAF50" if x >= mean_val else "#FF9800" for x in stats_sorted["Times Displayed"]]
    y_pos = range(len(stats_sorted))
    ax1.barh(y_pos, stats_sorted["Times Displayed"], color=colors, edgecolor="white")
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels([str(x)[:30] for x in stats_sorted["Item"]], fontsize=9)
    ax1.set_xlabel("Times Displayed")
    ax1.set_title("Display Frequency per Item")
    ax1.axvline(x=mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
    ax1.legend(loc="lower right")
    for i, (_idx, row) in enumerate(stats_sorted.iterrows()):
        ax1.text(
            row["Times Displayed"] + 0.5, i, f"{row['Times Displayed']:,}", va="center", fontsize=8
        )

    bar_height = 0.35
    y_pos_arr = np.arange(len(stats_sorted))
    ax2.barh(
        y_pos_arr - bar_height / 2,
        stats_sorted["Best Rate"] * 100,
        bar_height,
        label=f"% {pos_label}",
        color="#FFC000",
    )
    ax2.barh(
        y_pos_arr + bar_height / 2,
        stats_sorted["Worst Rate"] * 100,
        bar_height,
        label=f"% {neg_label}",
        color="#5B9BD5",
    )
    ax2.set_yticks(y_pos_arr)
    ax2.set_yticklabels([str(x)[:30] for x in stats_sorted["Item"]], fontsize=9)
    ax2.set_xlabel("Selection Rate (%)")
    ax2.set_title("Selection Rates per Item")
    ax2.legend(loc="lower right")
    ax2.set_xlim(0, 100)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_observed_percentages(
    df: pd.DataFrame, title: str, output_terms: tuple[str, str]
) -> Figure:
    """Stacked bar chart of % Best / % Unselected / % Worst per item."""
    pos_label, neg_label = output_terms
    pos_col = f"% Selected as {pos_label}"
    neg_col = f"% Selected as {neg_label}"
    df = df.sort_values("Score", ascending=True)

    fig = _new_figure(figsize=(12, max(4, len(df) * 0.4)))
    ax = fig.add_subplot(1, 1, 1)

    y_pos = range(len(df))
    ax.barh(y_pos, df[pos_col], color="#FFC000", label=pos_col)
    ax.barh(y_pos, df["% Unselected"], left=df[pos_col], color="#D9D9D9", label="% Unselected")
    ax.barh(
        y_pos, df[neg_col], left=df[pos_col] + df["% Unselected"], color="#5B9BD5", label=neg_col
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["Item"], fontsize=8)
    ax.set_title(title)
    ax.legend(loc="lower right", bbox_to_anchor=(1, -0.1), ncol=3)

    for i, (pos, unsel, neg) in enumerate(
        zip(df[pos_col], df["% Unselected"], df[neg_col], strict=True)
    ):
        if pos > 5:
            ax.text(pos / 2, i, f"{pos:.0f}%", va="center", ha="center", fontsize=8)
        if unsel > 5:
            ax.text(pos + unsel / 2, i, f"{unsel:.0f}%", va="center", ha="center", fontsize=8)
        if neg > 5:
            ax.text(100 - neg / 2, i, f"{neg:.0f}%", va="center", ha="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_scores(
    results: pd.DataFrame,
    positive_color: str,
    negative_color: str,
    error_bar_color: str,
    zero_line_color: str,
    anchor_item_color: str,
    anchor_item_error_color: str,
    title: str = "MaxDiff Scores",
    sample_size: int | None = None,
    segment_info: str | None = None,
    anchor_item: str | None = None,
    include_ci: bool = True,
) -> Figure:
    """Dot plot of per-item scores with optional CI whiskers."""
    sorted_results = results.sort_values("Score", ascending=True).reset_index(drop=True)
    num_items = len(sorted_results)

    fig = _new_figure(figsize=(13, max(0.4 * num_items, 10)), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    wrapped_labels = [textwrap.fill(str(label), width=50) for label in sorted_results["Item"]]
    has_ci = (
        include_ci
        and "Negative Error" in sorted_results.columns
        and "Positive Error" in sorted_results.columns
    )

    for i, row in sorted_results.iterrows():
        item = row["Item"]
        score = row["Score"]

        if item == anchor_item:
            point_color, error_color = anchor_item_color, anchor_item_error_color
        else:
            point_color = positive_color if score >= 0 else negative_color
            error_color = error_bar_color

        if has_ci:
            neg_err = row["Negative Error"]
            pos_err = row["Positive Error"]
            ax.errorbar(
                score,
                i,
                xerr=[[neg_err], [pos_err]],
                fmt="o",
                capsize=5,
                capthick=2,
                color=point_color,
                markersize=8,
                ecolor=error_color,
                elinewidth=2,
            )
        else:
            ax.plot(score, i, "o", color=point_color, markersize=8)

    if segment_info:
        title += f", {segment_info}"
    if sample_size:
        title += f" (n={sample_size})"

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(wrapped_labels, fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    ax.axvline(x=0, color=zero_line_color, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str) -> Figure:
    """Heatmap of item-by-item correlations.

    Drops the legacy seaborn dependency by using
    :meth:`matplotlib.axes.Axes.imshow` with a coolwarm diverging
    colormap. Produces visually equivalent output.
    """
    fig = _new_figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)

    data = corr_matrix.to_numpy()
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    n = len(corr_matrix.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index, rotation=0)
    # Light grid between cells to mimic seaborn linewidths=0.5.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Correlation")

    ax.set_title(title, fontsize=16)
    fig.tight_layout()
    return fig


def save_plot(fig: Figure, filename: str, base_dir: Path) -> None:
    """Save ``fig`` to ``<base_dir>/plots/{png,pdf}/<filename>.<ext>``.

    Creates the destination directories as needed. Does not call
    ``plt.close``; the caller's reference is the only one alive.
    """
    for ext in ("png", "pdf"):
        filepath = base_dir / "plots" / ext / f"{filename}.{ext}"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300 if ext == "png" else None, bbox_inches="tight")


def save_dataframe(
    df: pd.DataFrame, filename: str, base_dir: Path, include_index: bool = False
) -> None:
    """Save ``df`` to ``<base_dir>/data/{csv,xlsx}/<filename>.<ext>``."""
    for ext in ("csv", "xlsx"):
        filepath = base_dir / "data" / ext / f"{filename}.{ext}"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if ext == "csv":
            df.to_csv(filepath, index=include_index)
        else:
            df.to_excel(filepath, index=include_index)
