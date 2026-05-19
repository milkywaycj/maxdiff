"""Statistical correctness: Hierarchical Bayes utility recovery.

The HB MaxDiff model claims to recover the population utility vector
from observed best/worst choices. This test verifies that claim on
clean synthetic data drawn from a known utility vector.

Two recovery tests:

1. test_hb_recovers_utility_ordering - the items should be ranked in
   the same order as the truth. This is a weak claim and a high bar
   to fail.

2. test_hb_recovered_utilities_close_to_truth - after zero-centering,
   the recovered utilities should be within an absolute tolerance of
   the true utilities. This bites at numerical correctness, not just
   ordering.

A separate test in test_hb_bugs.py covers the *known buggy* behavior
(phantom item 0 padding, last-item-arbitrary-reference) by asserting
the correct behavior; those tests are xfail-strict until Phase 4
fixes the underlying bugs.

These tests skip automatically when numpyro is not installed (CI
runs them on Linux only; Windows skips). They are slow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tests.helpers import legacy_loader
from tests.helpers.synthetic_data import make_dataset

_HAS_NUMPYRO = (
    importlib.util.find_spec("numpyro") is not None and importlib.util.find_spec("jax") is not None
)
pytestmark = [
    pytest.mark.statistical,
    pytest.mark.hb,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_NUMPYRO, reason="numpyro/jax not installed"),
]


@pytest.fixture(scope="module")
def legacy():
    return legacy_loader.load_analyzer()


@pytest.fixture(scope="module")
def hb_fit():
    """Fit HB once on a moderate synthetic dataset and reuse across tests."""
    legacy = legacy_loader.load_analyzer()
    true_utilities = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    df = make_dataset(
        n_respondents=200,
        n_items=8,
        items_per_task=4,
        repeats_per_item=3,  # 6 tasks per respondent
        true_utilities=true_utilities,
        seed=4242,
    )
    attr_cols = [c for c in df.columns if c.startswith("Attribute")]

    model = legacy.HierarchicalBayesMaxDiff(
        n_iterations=400,
        n_warmup=400,
        n_chains=1,
        target_accept=0.9,
    )
    population = model.fit(df, attr_cols, "Most", "Least")
    return model, population, true_utilities, df


def test_hb_recovers_utility_ordering(hb_fit) -> None:
    """The Item column, sorted by Score descending, must match the
    item ordering implied by the true utility vector."""
    _model, population, true_utilities, _df = hb_fit
    # Truth ordering: items sorted by descending true utility.
    truth_order = np.argsort(-true_utilities)
    expected_labels = [f"Item {i + 1}" for i in truth_order]
    recovered_labels = population["Item"].tolist()
    assert recovered_labels == expected_labels, (
        f"HB did not recover the correct ordering.\n"
        f"  expected: {expected_labels}\n"
        f"  got:      {recovered_labels}"
    )


def test_hb_recovered_utilities_within_loose_tolerance(hb_fit) -> None:
    """Sanity check: after zero-centering, recovered utilities should
    be within a generous tolerance of the truth.

    The tolerance here (1.0) is loose because the current HB
    implementation has known biases that exaggerate extreme items
    (the reference-item parameterization issue, fix scheduled for
    Phase 4). The tighter
    ``test_hb_recovered_utilities_within_tight_tolerance_after_fix``
    test below pins the goal: post-Phase-4 the recovery should be
    within ~0.4. When that fix lands, both tests should be updated.
    """
    _model, population, true_utilities, _df = hb_fit

    recovered = dict(zip(population["Item"], population["Score"], strict=False))
    truth_labels = [f"Item {i + 1}" for i in range(len(true_utilities))]
    truth = dict(zip(truth_labels, true_utilities - true_utilities.mean(), strict=False))

    diffs = {label: recovered[label] - truth[label] for label in truth_labels}
    max_abs_diff = max(abs(d) for d in diffs.values())

    assert max_abs_diff < 1.0, (
        f"HB recovered utilities deviate from truth by up to {max_abs_diff:.3f}.\n"
        f"  per-item differences: {diffs}\n"
        f"  Even the loose tolerance is exceeded; a regression has occurred."
    )


@pytest.mark.xfail(
    reason="HB exaggerates extreme items due to the reference-item parameterization "
    "(last item arbitrary, prior asymmetric across items). Fix scheduled for Phase 4.",
    strict=True,
)
def test_hb_recovered_utilities_within_tight_tolerance_after_fix(hb_fit) -> None:
    """Goal: recovered utilities should be within 0.4 of truth across
    all items. Currently fails because the reference-item
    parameterization biases the extreme items outward. When Phase 4
    refactors the model to use a symmetric (sum-to-zero) constraint
    in the MCMC sampling itself, this test should turn green and the
    loose-tolerance counterpart can be tightened."""
    _model, population, true_utilities, _df = hb_fit

    recovered = dict(zip(population["Item"], population["Score"], strict=False))
    truth_labels = [f"Item {i + 1}" for i in range(len(true_utilities))]
    truth = dict(zip(truth_labels, true_utilities - true_utilities.mean(), strict=False))

    diffs = {label: recovered[label] - truth[label] for label in truth_labels}
    max_abs_diff = max(abs(d) for d in diffs.values())
    assert max_abs_diff < 0.4, (
        f"HB still deviates from truth by up to {max_abs_diff:.3f}.\n"
        f"  per-item differences: {diffs}"
    )


def test_hb_credible_intervals_not_completely_broken(hb_fit) -> None:
    """Regression guard: at least one credible interval should contain
    the truth.

    Currently HB produces CIs that frequently miss the truth because
    of the same reference-item bias that pulls extreme items outward.
    The post-Phase-4 goal of >=87% (7 of 8 with the standard 95%
    nominal coverage on 8 items) lives in the xfail-strict test
    below. This test only catches a total break (zero coverage)."""
    _model, population, true_utilities, _df = hb_fit

    truth = dict(
        zip(
            (f"Item {i + 1}" for i in range(len(true_utilities))),
            true_utilities - true_utilities.mean(),
            strict=False,
        )
    )

    covered = 0
    total = len(population)
    for _, row in population.iterrows():
        if row["2.5th Percentile"] <= truth[row["Item"]] <= row["97.5th Percentile"]:
            covered += 1

    assert covered >= 1, (
        f"HB credible intervals failed to cover ANY of {total} items - "
        f"the posterior summary is broken, not just biased."
    )


@pytest.mark.xfail(
    reason="HB credible intervals miss the truth on extreme items because of the "
    "reference-item parameterization bias. Fix scheduled for Phase 4.",
    strict=True,
)
def test_hb_credible_intervals_cover_truth_after_fix(hb_fit) -> None:
    """Goal: with N=200 and 8 items, at least 7 of 8 95% credible
    intervals should cover their true zero-centered utility.
    Currently fails for the same reasons as the tight-tolerance test
    above. Phase 4 should turn this green."""
    _model, population, true_utilities, _df = hb_fit

    truth = dict(
        zip(
            (f"Item {i + 1}" for i in range(len(true_utilities))),
            true_utilities - true_utilities.mean(),
            strict=False,
        )
    )

    covered = 0
    total = len(population)
    for _, row in population.iterrows():
        if row["2.5th Percentile"] <= truth[row["Item"]] <= row["97.5th Percentile"]:
            covered += 1

    assert covered >= total - 1, (
        f"HB covered only {covered}/{total} truth values within the 95% CI."
    )
