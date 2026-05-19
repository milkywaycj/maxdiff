"""Property-based invariants for the synthetic data generator.

The example-based tests in tests/unit/test_synthetic_data.py exercise
a handful of explicit parameter combinations. These Hypothesis tests
sweep a much wider space, looking for any combination of (n_items,
items_per_task, repeats_per_item, n_respondents, true_utilities, seed)
that violates an invariant the generator promises:

  P1. balanced design       - every item appears exactly
                              repeats_per_item times for every
                              respondent
  P2. distinct within task  - no item appears twice in the same task
  P3. choices are in the displayed set
  P4. best != worst         - within every task
  P5. determinism by seed   - identical inputs give identical outputs
  P6. correct row count     - n_respondents * tasks_per_respondent

Hypothesis will shrink any failing example to the smallest case that
still fails. If one of these properties ever breaks under refactor,
the failing seed appears in the CI log so the bug can be reproduced
deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.helpers.synthetic_data import make_dataset


def _feasible_design_params() -> st.SearchStrategy:
    """Strategy for (n_items, items_per_task, repeats_per_item) where
    n_items * repeats_per_item is divisible by items_per_task."""

    def _build(values: tuple[int, int, int]) -> tuple[int, int, int]:
        return values

    return st.builds(
        _build,
        st.tuples(
            st.integers(min_value=4, max_value=12),  # n_items
            st.integers(min_value=2, max_value=6),  # items_per_task
            st.integers(min_value=1, max_value=4),  # repeats_per_item
        ).filter(
            lambda triple: triple[1] <= triple[0] and (triple[0] * triple[2]) % triple[1] == 0
        ),
    )


@settings(
    max_examples=80,
    deadline=None,  # generator can be slow on harder designs; CI deadlines handled at job level
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    design=_feasible_design_params(),
    n_respondents=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_balanced_design_invariant(
    design: tuple[int, int, int], n_respondents: int, seed: int
) -> None:
    """P1 + P2 + P6 combined: every respondent sees a balanced design
    with no within-task duplicates and the right number of tasks."""
    n_items, items_per_task, repeats = design
    expected_tasks_per_resp = (n_items * repeats) // items_per_task

    df = make_dataset(
        n_respondents=n_respondents,
        n_items=n_items,
        items_per_task=items_per_task,
        repeats_per_item=repeats,
        true_utilities=np.zeros(n_items),
        seed=seed,
    )

    assert len(df) == n_respondents * expected_tasks_per_resp

    attr_cols = [c for c in df.columns if c.startswith("Attribute")]
    assert len(attr_cols) == items_per_task

    for resp_id, group in df.groupby("Response ID"):
        # Number of tasks for this respondent.
        assert len(group) == expected_tasks_per_resp
        # Within-task uniqueness.
        for _, row in group.iterrows():
            displayed = row[attr_cols].tolist()
            assert len(set(displayed)) == len(displayed), (
                f"Duplicate within task for {resp_id}: {displayed}"
            )
        # Per-respondent balance.
        counts = pd.Series(group[attr_cols].values.ravel()).value_counts()
        assert (counts == repeats).all(), f"Unbalanced design for {resp_id}: {counts.to_dict()}"


@settings(max_examples=60, deadline=None)
@given(
    design=_feasible_design_params(),
    n_respondents=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
    util_seed=st.integers(min_value=0, max_value=10_000),
)
def test_choices_well_formed(
    design: tuple[int, int, int], n_respondents: int, seed: int, util_seed: int
) -> None:
    """P3 + P4: Most and Least are both in the displayed set, and never equal."""
    n_items, items_per_task, repeats = design
    rng = np.random.default_rng(util_seed)
    utilities = rng.normal(size=n_items)

    df = make_dataset(
        n_respondents=n_respondents,
        n_items=n_items,
        items_per_task=items_per_task,
        repeats_per_item=repeats,
        true_utilities=utilities,
        seed=seed,
    )

    attr_cols = [c for c in df.columns if c.startswith("Attribute")]
    for _, row in df.iterrows():
        displayed = set(row[attr_cols].tolist())
        assert row["Most"] in displayed
        assert row["Least"] in displayed
        assert row["Most"] != row["Least"]


@settings(max_examples=30, deadline=None)
@given(
    design=_feasible_design_params(),
    n_respondents=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_seed_determinism(design: tuple[int, int, int], n_respondents: int, seed: int) -> None:
    """P5: identical inputs (including seed) produce identical DataFrames."""
    n_items, items_per_task, repeats = design
    kwargs = dict(
        n_respondents=n_respondents,
        n_items=n_items,
        items_per_task=items_per_task,
        repeats_per_item=repeats,
        true_utilities=np.zeros(n_items),
        seed=seed,
    )
    df_a = make_dataset(**kwargs)
    df_b = make_dataset(**kwargs)
    pd.testing.assert_frame_equal(df_a, df_b)
