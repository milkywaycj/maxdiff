"""Golden tests for the count-based analysis pipeline.

Each test:
  1. Builds a deterministic synthetic dataset (fixed seed, fixed
     true utility vector).
  2. Runs one analyzer function.
  3. Compares the output to a pinned CSV under ``expected/``.

Goldens guarantee that any refactor (Phase 3 extraction) or
intentional fix (Phase 4) makes the diff visible. Float comparisons
use a small tolerance to absorb harmless platform jitter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader
from tests.helpers.synthetic_data import make_dataset

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def legacy():
    return legacy_loader.load_analyzer()


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    return make_dataset(
        n_respondents=50,
        n_items=6,
        items_per_task=3,
        repeats_per_item=2,
        true_utilities=np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0]),
        seed=2024,
    )


@pytest.fixture(scope="module")
def medium_dataset() -> pd.DataFrame:
    return make_dataset(
        n_respondents=200,
        n_items=8,
        items_per_task=4,
        repeats_per_item=2,
        true_utilities=np.array([2.5, 1.5, 0.8, 0.2, -0.3, -1.0, -1.5, -2.2]),
        seed=20240519,
    )


# ----------------------------------------------------------------------
# calculate_scores_no_ci
# ----------------------------------------------------------------------


def test_scores_no_ci_small(legacy, small_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in small_dataset.columns if c.startswith("Attribute")]
    result = legacy.calculate_scores_no_ci(small_dataset, attr_cols, "Most", "Least")
    assert_matches_golden(result, "scores_no_ci_small.csv", sort_by=["Item"])


def test_scores_no_ci_medium(legacy, medium_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in medium_dataset.columns if c.startswith("Attribute")]
    result = legacy.calculate_scores_no_ci(medium_dataset, attr_cols, "Most", "Least")
    assert_matches_golden(result, "scores_no_ci_medium.csv", sort_by=["Item"])


# ----------------------------------------------------------------------
# calculate_observed_percentages
# ----------------------------------------------------------------------


def test_observed_percentages_small(legacy, small_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in small_dataset.columns if c.startswith("Attribute")]
    result = legacy.calculate_observed_percentages(
        small_dataset, attr_cols, "Most", "Least", ("Best", "Worst")
    )
    assert_matches_golden(result, "observed_percentages_small.csv", sort_by=["Item"])


def test_observed_percentages_medium(legacy, medium_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in medium_dataset.columns if c.startswith("Attribute")]
    result = legacy.calculate_observed_percentages(
        medium_dataset, attr_cols, "Most", "Least", ("Best", "Worst")
    )
    assert_matches_golden(result, "observed_percentages_medium.csv", sort_by=["Item"])


# ----------------------------------------------------------------------
# calculate_display_statistics
# ----------------------------------------------------------------------


def test_display_statistics_small(legacy, small_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in small_dataset.columns if c.startswith("Attribute")]
    stats_df, _balance = legacy.calculate_display_statistics(
        small_dataset, attr_cols, "Most", "Least"
    )
    assert_matches_golden(stats_df, "display_statistics_small.csv", sort_by=["Item"])


def test_display_statistics_medium(legacy, medium_dataset, assert_matches_golden) -> None:
    attr_cols = [c for c in medium_dataset.columns if c.startswith("Attribute")]
    stats_df, _balance = legacy.calculate_display_statistics(
        medium_dataset, attr_cols, "Most", "Least"
    )
    assert_matches_golden(stats_df, "display_statistics_medium.csv", sort_by=["Item"])


# ----------------------------------------------------------------------
# bootstrap_analysis - small fixed iterations for determinism
# ----------------------------------------------------------------------


def test_bootstrap_small_fixed_seed(legacy, small_dataset, assert_matches_golden) -> None:
    """Bootstrap output depends on numpy's default_rng() default seed,
    which is non-deterministic. To make this golden reproducible we
    seed numpy globally before running."""
    np.random.seed(424242)
    attr_cols = [c for c in small_dataset.columns if c.startswith("Attribute")]
    unique_attrs = pd.unique(small_dataset[attr_cols].values.ravel("K"))
    unique_attrs = unique_attrs[~pd.isnull(unique_attrs)]
    attr_to_index = {a: i for i, a in enumerate(unique_attrs)}
    result = legacy.bootstrap_analysis(
        small_dataset,
        attr_cols,
        unique_attrs,
        attr_to_index,
        "Most",
        "Least",
        n_iterations=500,
    )
    # Bootstrap is non-deterministic across numpy releases. The Score
    # column (observed) is deterministic; percentile columns can
    # vary slightly. Compare only the deterministic columns.
    deterministic = result[["Item", "Score"]]
    assert_matches_golden(deterministic, "bootstrap_observed_scores_small.csv", sort_by=["Item"])
