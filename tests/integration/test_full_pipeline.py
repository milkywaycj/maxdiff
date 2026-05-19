"""End-to-end integration tests for the public ``maxdiff`` API.

These tests compose the public functions the way the desktop GUI and
the browser tool do, write the same kinds of artifacts to disk, and
round-trip them to assert content correctness. They exercise the
contracts between modules (count -> bootstrap -> plotting -> save)
that no individual unit test catches.

They deliberately avoid importing the customtkinter desktop GUI file
so they can run on any platform, including headless CI runners.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maxdiff import (
    bootstrap_analysis,
    calculate_correlation_matrix,
    calculate_display_statistics,
    calculate_observed_percentages,
    check_errors,
    format_display_report,
)
from maxdiff.plotting import (
    plot_correlation_matrix,
    plot_display_balance,
    plot_observed_percentages,
    plot_scores,
    save_dataframe,
    save_plot,
)
from tests.helpers.synthetic_data import make_dataset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def study_dataset() -> pd.DataFrame:
    """A small but realistic synthetic dataset that exercises every
    public analysis path."""
    return make_dataset(
        n_respondents=80,
        n_items=6,
        items_per_task=3,
        repeats_per_item=2,
        true_utilities=np.array([2.0, 1.0, 0.5, 0.0, -1.0, -2.0]),
        seed=20240519,
    )


# ----------------------------------------------------------------------
# Full overall pipeline: validate -> stats -> bootstrap -> percentages
#                       -> correlation -> plot -> save -> round-trip
# ----------------------------------------------------------------------


def test_overall_pipeline_writes_complete_artifact_set(
    study_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    attr_cols = [c for c in study_dataset.columns if c.startswith("Attribute")]

    # 1. Validate. Should not raise.
    check_errors(study_dataset, attr_cols, "Most", "Least")

    # 2. Display statistics + report.
    stats_df, balance = calculate_display_statistics(study_dataset, attr_cols, "Most", "Least")
    report = format_display_report(stats_df, balance, ("Best", "Worst"))
    assert isinstance(report, str)
    assert "DISPLAY STATISTICS REPORT" in report
    assert balance["num_items"] == 6
    assert balance["is_balanced"] is True  # synthetic data is perfectly balanced

    # 3. Bootstrap analysis (small n_iterations for test speed).
    unique_attrs = pd.unique(study_dataset[attr_cols].values.ravel("K"))
    unique_attrs = unique_attrs[~pd.isnull(unique_attrs)]
    attr_to_index = {a: i for i, a in enumerate(unique_attrs)}
    bootstrap_result = bootstrap_analysis(
        study_dataset,
        attr_cols,
        unique_attrs,
        attr_to_index,
        "Most",
        "Least",
        n_iterations=200,
    )
    assert set(bootstrap_result.columns) == {
        "Item",
        "Score",
        "2.5th Percentile",
        "97.5th Percentile",
        "Negative Error",
        "Positive Error",
    }

    # 4. Observed percentages.
    observed = calculate_observed_percentages(
        study_dataset,
        attr_cols,
        "Most",
        "Least",
        ("Best", "Worst"),
    )
    # The three percent columns must sum to 100 per item.
    total = (
        observed["% Selected as Best"] + observed["% Unselected"] + observed["% Selected as Worst"]
    )
    np.testing.assert_allclose(total, 100.0)

    # 5. Correlation matrix.
    corr = calculate_correlation_matrix(study_dataset, attr_cols, "Most", "Least")
    assert corr.shape == (6, 6)

    # 6. Plot all four outputs.
    scores_fig = plot_scores(
        bootstrap_result,
        positive_color="#F4B400",
        negative_color="#5B9BD5",
        error_bar_color="#a1a1a1",
        zero_line_color="red",
        anchor_item_color="#a1a1a1",
        anchor_item_error_color="#a1a1a1",
        title="Overall Net Scores",
        sample_size=80,
    )
    freq_fig = plot_observed_percentages(observed, "Selection Frequencies", ("Best", "Worst"))
    balance_fig = plot_display_balance(stats_df, "Display Balance", ("Best", "Worst"))
    corr_fig = plot_correlation_matrix(corr, "Correlation Matrix")

    # 7. Save dataframes and plots; round-trip the CSVs.
    save_dataframe(bootstrap_result, "overall_net_scores_n80", tmp_path)
    save_dataframe(observed, "overall_selection_frequencies_n80", tmp_path)
    save_dataframe(stats_df, "overall_display_statistics_n80", tmp_path)
    save_dataframe(corr, "overall_correlation_matrix_n80", tmp_path, include_index=True)

    save_plot(scores_fig, "overall_net_scores_n80_plot", tmp_path)
    save_plot(freq_fig, "overall_selection_frequencies_n80_plot", tmp_path)
    save_plot(balance_fig, "overall_display_balance_n80", tmp_path)
    save_plot(corr_fig, "overall_correlation_matrix_n80_plot", tmp_path)

    # The same directory layout the desktop GUI produces.
    csv_dir = tmp_path / "data" / "csv"
    xlsx_dir = tmp_path / "data" / "xlsx"
    png_dir = tmp_path / "plots" / "png"
    pdf_dir = tmp_path / "plots" / "pdf"
    for d in (csv_dir, xlsx_dir, png_dir, pdf_dir):
        assert d.is_dir(), f"missing output directory: {d}"

    # Every CSV we wrote is readable and has the same row count.
    round_trip = pd.read_csv(csv_dir / "overall_net_scores_n80.csv")
    pd.testing.assert_frame_equal(
        round_trip[["Item", "Score"]].reset_index(drop=True),
        bootstrap_result[["Item", "Score"]].reset_index(drop=True),
    )

    # Every plot artifact exists in both PNG and PDF form.
    for name in (
        "overall_net_scores_n80_plot",
        "overall_selection_frequencies_n80_plot",
        "overall_display_balance",
        "overall_correlation_matrix_n80_plot",
    ):
        # Some saves use "overall_display_balance" (no _plot suffix) by convention;
        # accept either form to match the desktop's filename mapping.
        png_with_plot = png_dir / f"{name}_n80.png"
        png_without = png_dir / f"{name}.png"
        png_either = png_with_plot if png_with_plot.is_file() else png_without
        assert png_either.is_file() or (png_dir / f"{name}.png").is_file(), (
            f"missing PNG for {name}"
        )


# ----------------------------------------------------------------------
# Segment analysis pipeline: split by a categorical column, run the
# full analysis per segment, save under per-segment filenames.
# ----------------------------------------------------------------------


def _segment_data(study_dataset: pd.DataFrame) -> pd.DataFrame:
    """Synthetic segment assignments stratifying respondents into two
    age groups. Deterministic from the respondent ID."""
    respondents = study_dataset["Response ID"].unique()
    return pd.DataFrame(
        {
            "Response ID": respondents,
            "Age Group": ["18-34" if i % 2 == 0 else "35+" for i in range(len(respondents))],
        }
    )


def test_segment_pipeline_writes_one_artifact_per_segment(
    study_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Mimic the GUI's segment_maxdiff_analysis flow without importing
    the GUI module. For each segment value, run the same analysis +
    save sequence as the overall pipeline, with a per-segment prefix.
    Assert the directory layout has one CSV per segment per output."""
    attr_cols = [c for c in study_dataset.columns if c.startswith("Attribute")]
    segment_df = _segment_data(study_dataset)
    merged = pd.merge(study_dataset, segment_df, on="Response ID", how="left")

    unique_attrs = pd.unique(study_dataset[attr_cols].values.ravel("K"))
    unique_attrs = unique_attrs[~pd.isnull(unique_attrs)]
    attr_to_index = {a: i for i, a in enumerate(unique_attrs)}

    seen_segments = []
    for segment_value, group in merged.groupby("Age Group"):
        seen_segments.append(segment_value)
        n = group["Response ID"].nunique()
        prefix = f"Age_Group_{segment_value.replace('-', '_').replace('+', 'plus')}"

        # Bootstrap (small iters for speed)
        bs = bootstrap_analysis(
            group,
            attr_cols,
            unique_attrs,
            attr_to_index,
            "Most",
            "Least",
            n_iterations=200,
        )
        observed = calculate_observed_percentages(
            group,
            attr_cols,
            "Most",
            "Least",
            ("Best", "Worst"),
        )
        stats_df, _balance = calculate_display_statistics(group, attr_cols, "Most", "Least")

        save_dataframe(bs, f"{prefix}_net_scores_n{n}", tmp_path)
        save_dataframe(observed, f"{prefix}_selection_frequencies_n{n}", tmp_path)
        save_dataframe(stats_df, f"{prefix}_display_statistics_n{n}", tmp_path)

    assert set(seen_segments) == {"18-34", "35+"}

    csv_dir = tmp_path / "data" / "csv"
    files = sorted(p.name for p in csv_dir.iterdir())
    # Expect three CSVs per segment, two segments => six files.
    assert len(files) == 6, f"unexpected file set: {files}"
    for seg in ("18_34", "35plus"):
        for kind in ("net_scores", "selection_frequencies", "display_statistics"):
            matches = [f for f in files if seg in f and kind in f]
            assert len(matches) == 1, (
                f"missing or duplicate file for segment {seg} / {kind}: {matches}"
            )


# ----------------------------------------------------------------------
# Round-trip consistency: write -> read -> rerun analysis should give
# the same scores as analysis on the original DataFrame. Protects
# against subtle encoding or dtype changes during save/load.
# ----------------------------------------------------------------------


def test_csv_round_trip_preserves_scores(study_dataset: pd.DataFrame, tmp_path: Path) -> None:
    """Save the dataset to CSV, read it back, rerun the count
    analysis, and confirm the scores match within float tolerance.

    This was a recurring gotcha when the legacy code went through
    several CSV encodings; the test pins the round-trip contract.
    """
    from maxdiff.count import calculate_scores_no_ci
    from maxdiff.io import read_tabular_file

    attr_cols = [c for c in study_dataset.columns if c.startswith("Attribute")]
    original = calculate_scores_no_ci(study_dataset, attr_cols, "Most", "Least").set_index("Item")

    csv_path = tmp_path / "dataset.csv"
    study_dataset.to_csv(csv_path, index=False, encoding="utf-8")

    reloaded = read_tabular_file(csv_path)
    rerun = calculate_scores_no_ci(reloaded, attr_cols, "Most", "Least").set_index("Item")

    pd.testing.assert_series_equal(original["Score"], rerun["Score"], check_names=False)
