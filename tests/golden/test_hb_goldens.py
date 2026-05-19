"""Golden test for Hierarchical Bayes population output.

The HB MaxDiff estimator is the project's most complex numerical
pipeline and the most expensive to test from scratch. The
`tests/statistical/test_hb_recovery.py` suite already verifies that the
model can recover a known utility vector on a moderate sample — that
is the *correctness* test. This module is the complementary
*regression* test: it pins the population output of a small, fast,
fixed-seed fit so any future refactor that quietly shifts HB results
fails CI rather than silently re-baselining downstream consumers.

Design constraints:

* **Small and fast.** N=100 respondents, 500 warmup + 500 samples,
  one chain. Runs in ~15-30s on a developer laptop. Statistical
  power for *recovery* is poor at this size, which is fine — the
  golden tests pin "what HB currently produces on this fixture",
  not "what HB ideally produces".
* **Generous tolerance.** ``float_tol=0.1`` is well outside MCMC
  sampling noise on the same seed and JAX version, but loose enough
  to absorb the small numerical drift that occurs when JAX itself
  changes. Real algorithmic regressions in HB shift results far more
  than that. If the tolerance ever needs to be regenerated for a
  major JAX bump, rerun with ``pytest --update-goldens`` on the
  same Linux JAX version CI uses, and commit the new fixture
  alongside the JAX version bump.
* **Pinned columns only.** The Item ordering and the Score column are
  the lowest-variance outputs from a small-sample HB fit. The
  percentile and error columns vary more than the Score does and are
  not pinned here; the statistical recovery test exercises that path.
* **Marked slow + hb + golden.** Runs on Linux CI only (same as the
  rest of the HB suite) and skips automatically when numpyro/jax is
  not available.
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
    pytest.mark.golden,
    pytest.mark.hb,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_NUMPYRO, reason="numpyro/jax not installed"),
]


@pytest.fixture(scope="module")
def hb_small_fit():
    """Fit HB once on a small fixed-seed dataset and reuse across tests.

    The fixture is intentionally small to keep the test fast. N=100
    is too few respondents to recover utilities accurately, but it
    *is* enough to reliably pin a regression-detection golden: the
    same seed + the same JAX install produces the same numbers each
    run, modulo sub-millivolt floating-point drift.
    """
    legacy = legacy_loader.load_analyzer()

    true_utilities = np.array([1.5, 0.8, 0.0, -0.8, -1.5])
    df = make_dataset(
        n_respondents=100,
        n_items=5,
        items_per_task=3,
        repeats_per_item=3,  # 5 tasks per respondent
        true_utilities=true_utilities,
        seed=2026,
    )
    attr_cols = [c for c in df.columns if c.startswith("Attribute")]

    model = legacy.HierarchicalBayesMaxDiff(
        n_iterations=500,
        n_warmup=500,
        n_chains=1,
        target_accept=0.9,
    )
    population = model.fit(df, attr_cols, "Most", "Least")
    return population


def test_hb_population_scores_golden(hb_small_fit, assert_matches_golden) -> None:
    """Pin the Item ordering and Score column from a small HB fit.

    Tolerance is generous (``float_tol=0.1``) to absorb harmless
    JAX-version drift while still catching any change that
    meaningfully shifts HB output. The percentile and error columns
    are deliberately omitted from the comparison — they have higher
    posterior variance at this sample size and are exercised by
    ``tests/statistical/test_hb_recovery.py``.
    """
    pinned = hb_small_fit[["Item", "Score"]]
    assert_matches_golden(
        pinned,
        "hb_population_scores_small.csv",
        sort_by=["Item"],
        float_tol=0.1,
    )


def test_hb_credible_interval_widths_golden(hb_small_fit, assert_matches_golden) -> None:
    """Pin the credible-interval widths (upper - lower) per item.

    CI widths are *much* more stable than the absolute percentile
    locations because they cancel the posterior's overall location
    drift. A regression that broadens or collapses HB's uncertainty
    estimates would fail this even when the Score golden passes.
    Tolerance ``float_tol=0.15`` allows for JAX/numerical drift in
    the percentile estimates themselves while still catching gross
    changes in uncertainty calibration.
    """
    widths = hb_small_fit.assign(
        CI_Width=hb_small_fit["97.5th Percentile"] - hb_small_fit["2.5th Percentile"],
    )[["Item", "CI_Width"]]
    assert_matches_golden(
        widths,
        "hb_credible_interval_widths_small.csv",
        sort_by=["Item"],
        float_tol=0.15,
    )
