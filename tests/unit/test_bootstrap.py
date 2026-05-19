"""Characterization tests for bootstrap_analysis.

The bootstrap path produces (Score, 2.5th Percentile, 97.5th Percentile,
Negative Error, Positive Error) by resampling respondents with
replacement and recomputing scores per draw. These tests pin the
shape and signature contract, the deterministic relationship between
the columns, and the bounding-box invariant that the point estimate
must lie inside the credible interval.

Bootstrap *coverage* (does the 95% CI cover the true score 95% of the
time?) is a statistical test belonging in tests/statistical/. Here
we cover what can be asserted without Monte Carlo verification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader
from tests.helpers.synthetic_data import make_dataset


@pytest.fixture(scope="module")
def legacy():
    return legacy_loader.load_analyzer()


@pytest.fixture(scope="module")
def sample_dataset() -> pd.DataFrame:
    """A medium-size synthetic dataset re-used across tests in this
    module. Module-scoped so the choice simulation runs once."""
    return make_dataset(
        n_respondents=80,
        n_items=8,
        items_per_task=4,
        repeats_per_item=2,
        true_utilities=np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
        seed=2024,
    )


def _attr_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Attribute")]


def _unique_attrs(df: pd.DataFrame) -> np.ndarray:
    attrs = pd.unique(df[_attr_cols(df)].values.ravel("K"))
    return attrs[~pd.isnull(attrs)]


def _run_bootstrap(legacy_module, df: pd.DataFrame, n_iterations: int = 200) -> pd.DataFrame:
    unique_attrs = _unique_attrs(df)
    attr_to_index = {attr: i for i, attr in enumerate(unique_attrs)}
    return legacy_module.bootstrap_analysis(
        df,
        _attr_cols(df),
        unique_attrs,
        attr_to_index,
        "Most",
        "Least",
        n_iterations=n_iterations,
    )


class TestBootstrapShape:
    def test_returns_dataframe_with_expected_columns(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=100)
        for col in [
            "Item",
            "Score",
            "2.5th Percentile",
            "97.5th Percentile",
            "Negative Error",
            "Positive Error",
        ]:
            assert col in result.columns

    def test_one_row_per_item(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=100)
        assert len(result) == len(_unique_attrs(sample_dataset))

    def test_results_sorted_by_score_descending(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=100)
        scores = result["Score"].tolist()
        assert scores == sorted(scores, reverse=True)


class TestBootstrapInvariants:
    """Invariants the percentile error columns must satisfy under any
    valid input."""

    def test_lower_percentile_is_at_most_score(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=200)
        # The 2.5th percentile of the bootstrap distribution may be
        # below the observed score, but with finite samples it can
        # occasionally exceed it by a tiny epsilon for tightly-clustered
        # items. Allow a small numerical tolerance.
        violations = result[result["2.5th Percentile"] > result["Score"] + 1e-6]
        assert violations.empty, f"2.5th percentile above Score for: {violations}"

    def test_upper_percentile_is_at_least_score(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=200)
        violations = result[result["97.5th Percentile"] < result["Score"] - 1e-6]
        assert violations.empty, f"97.5th percentile below Score for: {violations}"

    def test_negative_error_equals_score_minus_lower(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=200)
        expected = result["Score"] - result["2.5th Percentile"]
        np.testing.assert_allclose(result["Negative Error"], expected)

    def test_positive_error_equals_upper_minus_score(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=200)
        expected = result["97.5th Percentile"] - result["Score"]
        np.testing.assert_allclose(result["Positive Error"], expected)

    def test_lower_at_most_upper(self, legacy, sample_dataset) -> None:
        result = _run_bootstrap(legacy, sample_dataset, n_iterations=200)
        assert (result["2.5th Percentile"] <= result["97.5th Percentile"]).all()


class TestBootstrapPointEstimateMatchesObserved:
    """The Score column of bootstrap_analysis should be the *observed*
    score on the original sample, not the bootstrap mean. This is the
    convention used by the analyzer; pinning it here."""

    def test_score_matches_calculate_scores_no_ci(self, legacy, sample_dataset) -> None:
        bootstrap_result = _run_bootstrap(legacy, sample_dataset, n_iterations=200).set_index(
            "Item"
        )
        count_result = legacy.calculate_scores_no_ci(
            sample_dataset, _attr_cols(sample_dataset), "Most", "Least"
        ).set_index("Item")
        # Same items, same Score column values.
        for item in count_result.index:
            assert bootstrap_result.loc[item, "Score"] == pytest.approx(
                count_result.loc[item, "Score"]
            )


class TestBootstrapTightensWithMoreData:
    """A coarse property: confidence intervals should shrink (in
    expectation) as the sample size grows. Used here as a sanity test
    rather than a formal coverage check."""

    @pytest.mark.slow
    def test_ci_width_tightens_with_more_respondents(self, legacy) -> None:
        utilities = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])

        small = make_dataset(
            n_respondents=30,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=utilities,
            seed=11,
        )
        large = make_dataset(
            n_respondents=300,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=utilities,
            seed=11,
        )

        small_result = _run_bootstrap(legacy, small, n_iterations=400)
        large_result = _run_bootstrap(legacy, large, n_iterations=400)

        small_width = (small_result["97.5th Percentile"] - small_result["2.5th Percentile"]).mean()
        large_width = (large_result["97.5th Percentile"] - large_result["2.5th Percentile"]).mean()

        # Width should approximately halve as sample size grows ~10x;
        # use a soft factor of 2x as the assertion threshold.
        assert large_width < small_width * 0.6, (
            f"CI did not tighten as expected: small={small_width:.2f}, large={large_width:.2f}"
        )
