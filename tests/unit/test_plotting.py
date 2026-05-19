"""Tests for the thread-safe plotting helpers.

These tests do not assert pixel-perfect outputs (matplotlib version
drift makes that brittle). They cover the contract callers rely on:

  * Each plot function returns a ``matplotlib.figure.Figure``.
  * No call touches the pyplot global state, so calling from a
    worker thread is safe.
  * ``save_plot`` produces both PNG and PDF on disk under the
    expected directory layout.
  * ``save_dataframe`` writes both CSV and XLSX.

A focused thread-safety test starts multiple worker threads that
each create and save a figure, and asserts no exceptions surface.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from maxdiff.plotting import (
    plot_correlation_matrix,
    plot_display_balance,
    plot_observed_percentages,
    plot_scores,
    save_dataframe,
    save_plot,
)


@pytest.fixture
def scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item": [f"Item {i}" for i in range(1, 6)],
            "Score": [50.0, 25.0, 0.0, -25.0, -50.0],
            "Negative Error": [5.0, 4.0, 3.0, 4.0, 5.0],
            "Positive Error": [5.0, 4.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def display_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item": [f"Item {i}" for i in range(1, 6)],
            "Times Displayed": [120, 119, 121, 120, 120],
            "Times Selected Best": [30, 25, 20, 15, 10],
            "Times Selected Worst": [10, 15, 20, 25, 30],
            "Times Unselected": [80, 79, 81, 80, 80],
            "Best Rate": [0.25, 0.21, 0.165, 0.125, 0.083],
            "Worst Rate": [0.083, 0.126, 0.165, 0.208, 0.25],
        }
    )


@pytest.fixture
def observed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item": [f"Item {i}" for i in range(1, 4)],
            "% Selected as Best": [40.0, 30.0, 10.0],
            "% Unselected": [50.0, 50.0, 50.0],
            "% Selected as Worst": [10.0, 20.0, 40.0],
            "Score": [30.0, 10.0, -30.0],
        }
    )


# ----------------------------------------------------------------------
# Return-type contract
# ----------------------------------------------------------------------


def test_plot_scores_returns_figure(scores_df) -> None:
    fig = plot_scores(
        scores_df,
        positive_color="#F4B400",
        negative_color="#F4B400",
        error_bar_color="#a1a1a1",
        zero_line_color="red",
        anchor_item_color="#a1a1a1",
        anchor_item_error_color="#a1a1a1",
    )
    assert isinstance(fig, Figure)


def test_plot_observed_percentages_returns_figure(observed_df) -> None:
    fig = plot_observed_percentages(observed_df, "Test", ("Best", "Worst"))
    assert isinstance(fig, Figure)


def test_plot_display_balance_returns_figure(display_stats_df) -> None:
    fig = plot_display_balance(display_stats_df)
    assert isinstance(fig, Figure)


def test_plot_correlation_matrix_returns_figure() -> None:
    items = ["A", "B", "C"]
    corr = pd.DataFrame(np.eye(3), index=items, columns=items)
    fig = plot_correlation_matrix(corr, "Corr")
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# Thread safety - the key Phase 6c claim
# ----------------------------------------------------------------------


def test_plot_functions_safe_to_call_from_worker_threads(
    scores_df, display_stats_df, observed_df, tmp_path
) -> None:
    """Run plot_scores / plot_observed_percentages / plot_display_balance
    in parallel from 4 worker threads. The Phase 6c rewrite to the OO
    API means no shared pyplot state is touched and concurrent
    invocations are independent.

    Pre-Phase-6c code used plt.subplots and plt.close in the same
    functions, which under high enough concurrency could deadlock on
    the matplotlib font cache lock or leak figures.
    """
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            f1 = plot_scores(
                scores_df,
                positive_color="#F4B400",
                negative_color="#F4B400",
                error_bar_color="#a1a1a1",
                zero_line_color="red",
                anchor_item_color="#a1a1a1",
                anchor_item_error_color="#a1a1a1",
                title=f"Thread {i}",
            )
            f2 = plot_observed_percentages(observed_df, f"Thread {i}", ("Best", "Worst"))
            f3 = plot_display_balance(display_stats_df, title=f"Thread {i}")
            save_plot(f1, f"scores_{i}", tmp_path)
            save_plot(f2, f"freq_{i}", tmp_path)
            save_plot(f3, f"balance_{i}", tmp_path)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Worker threads raised: {errors}"


# ----------------------------------------------------------------------
# Save helpers
# ----------------------------------------------------------------------


def test_save_plot_writes_png_and_pdf(scores_df, tmp_path: Path) -> None:
    fig = plot_scores(
        scores_df,
        positive_color="#F4B400",
        negative_color="#F4B400",
        error_bar_color="#a1a1a1",
        zero_line_color="red",
        anchor_item_color="#a1a1a1",
        anchor_item_error_color="#a1a1a1",
    )
    save_plot(fig, "test_scores", tmp_path)
    assert (tmp_path / "plots" / "png" / "test_scores.png").is_file()
    assert (tmp_path / "plots" / "pdf" / "test_scores.pdf").is_file()


def test_save_dataframe_writes_csv_and_xlsx(scores_df, tmp_path: Path) -> None:
    save_dataframe(scores_df, "test_df", tmp_path)
    csv_path = tmp_path / "data" / "csv" / "test_df.csv"
    xlsx_path = tmp_path / "data" / "xlsx" / "test_df.xlsx"
    assert csv_path.is_file()
    assert xlsx_path.is_file()
    # Round-trip the CSV to confirm we wrote the expected columns.
    round_tripped = pd.read_csv(csv_path)
    assert list(round_tripped.columns) == list(scores_df.columns)
