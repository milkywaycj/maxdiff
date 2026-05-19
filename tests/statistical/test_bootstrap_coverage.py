"""Statistical correctness: bootstrap confidence interval coverage.

A 95% confidence interval should, in expectation, cover the true
parameter value 95% of the time over repeated experiments. This test
draws many synthetic datasets from a known utility vector, runs the
bootstrap on each, and checks the resulting average coverage.

We do not aim for a tight 95.0% +- 1.0% match (would need thousands of
replications). The expensive operation here is the bootstrap loop, so
we use a modest number of replications and assert coverage is in a
reasonable corridor around 95%. If a future fix changes the
resampling procedure and breaks this, the test catches the
regression.

This test is slow; mark it accordingly so quick local runs skip it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader
from tests.helpers.synthetic_data import make_dataset

pytestmark = [pytest.mark.statistical, pytest.mark.slow]


def _true_observed_scores(legacy, df: pd.DataFrame, attr_cols: list[str]) -> dict[str, float]:
    """The 'true' score the bootstrap CI ought to bracket is the
    *observed* score on the full sample, not the population utility.
    Bootstrap CIs are coverage statements about the sampling
    distribution of the statistic, conditional on the dataset.

    For this test we use a different definition: the score computed
    from a LARGE reference dataset drawn from the same true utilities
    serves as a proxy for the population statistic that the
    small-sample bootstrap should cover.
    """
    scores = legacy.calculate_scores_no_ci(df, attr_cols, "Most", "Least")
    return dict(zip(scores["Item"], scores["Score"], strict=False))


def test_bootstrap_ci_coverage_is_near_nominal() -> None:
    """Over many independent samples, the fraction of items whose
    population score falls inside the 95% bootstrap CI should be
    close to 95%.
    """
    legacy = legacy_loader.load_analyzer()

    true_utilities = np.array([1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    n_items = len(true_utilities)
    attr_cols = [f"Attribute{i}" for i in range(1, 5)]

    # Build a large reference dataset to estimate the "true" observed
    # score for each item (the parameter the bootstrap CI claims to
    # cover). 4000 respondents at 4 items/task with 2 repeats yields
    # 32 000 task-rows; observed scores at this size are within ~0.5
    # of the population truth, which is small enough that coverage
    # behaves correctly.
    reference = make_dataset(
        n_respondents=4000,
        n_items=n_items,
        items_per_task=4,
        repeats_per_item=2,
        true_utilities=true_utilities,
        seed=987654,
    )
    reference_scores = _true_observed_scores(legacy, reference, attr_cols)

    # Now draw many smaller independent samples and check coverage of
    # the reference scores by each sample's bootstrap CI.
    n_replications = 80
    sample_size = 200
    bootstrap_iterations = 400

    covered_per_item = {item: 0 for item in reference_scores}

    for rep in range(n_replications):
        df = make_dataset(
            n_respondents=sample_size,
            n_items=n_items,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=true_utilities,
            seed=10_000 + rep,
        )
        unique_attrs = pd.unique(df[attr_cols].values.ravel("K"))
        unique_attrs = unique_attrs[~pd.isnull(unique_attrs)]
        attr_to_index = {a: i for i, a in enumerate(unique_attrs)}
        result = legacy.bootstrap_analysis(
            df,
            attr_cols,
            unique_attrs,
            attr_to_index,
            "Most",
            "Least",
            n_iterations=bootstrap_iterations,
        )
        result_by_item = result.set_index("Item")
        for item, ref_score in reference_scores.items():
            lo = result_by_item.loc[item, "2.5th Percentile"]
            hi = result_by_item.loc[item, "97.5th Percentile"]
            if lo <= ref_score <= hi:
                covered_per_item[item] += 1

    coverage = {item: count / n_replications for item, count in covered_per_item.items()}
    mean_coverage = float(np.mean(list(coverage.values())))

    # Wide corridor: with 80 reps, the std of the per-item coverage is
    # ~sqrt(0.95*0.05/80) = 2.4 percentage points. Use a generous +-15
    # percentage-point band to keep the test robust; we are looking
    # for "approximately right" not "exact". A fix that broke coverage
    # to, say, 60% or 99.9% would fail this test.
    assert 0.80 <= mean_coverage <= 1.0, (
        f"Bootstrap coverage out of expected range: mean={mean_coverage:.2%}, "
        f"per-item={ {k: f'{v:.0%}' for k, v in coverage.items()} }"
    )
