"""Tests for the synthetic MaxDiff data generator.

Written FIRST (red), then the generator was implemented to satisfy
these tests (green). Most of the test suite, especially the
statistical-recovery tests in Phase 2, builds on this generator, so
its correctness is load-bearing. The tests here cover:

* shape and schema of the produced DataFrame
* the balanced-design invariant (every item appears exactly
  repeats_per_item times per respondent across the tasks they see)
* the well-formedness invariant (the items selected as Most/Least
  are always among the items displayed in that task)
* reproducibility (same seed -> identical output)
* sensitivity to seed (different seeds -> different output) and to
  the true utility vector (higher-utility items get picked more)
* input validation (impossible designs raise a clear error)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers.synthetic_data import (
    SyntheticDesignError,
    make_dataset,
)

# ----------------------------------------------------------------------
# Shape / schema
# ----------------------------------------------------------------------


class TestShape:
    def test_returns_dataframe(self) -> None:
        df = make_dataset(
            n_respondents=5,
            n_items=6,
            items_per_task=3,
            repeats_per_item=2,
            true_utilities=np.zeros(6),
            seed=0,
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        df = make_dataset(
            n_respondents=5,
            n_items=6,
            items_per_task=3,
            repeats_per_item=2,
            true_utilities=np.zeros(6),
            seed=0,
        )
        # Response ID, Attribute1..Attribute3, Most, Least
        expected = ["Response ID", "Attribute1", "Attribute2", "Attribute3", "Most", "Least"]
        assert list(df.columns) == expected

    def test_row_count_matches_respondents_times_tasks(self) -> None:
        # n_items * repeats / items_per_task tasks per respondent
        # 6 * 2 / 3 = 4 tasks per respondent
        df = make_dataset(
            n_respondents=5,
            n_items=6,
            items_per_task=3,
            repeats_per_item=2,
            true_utilities=np.zeros(6),
            seed=0,
        )
        assert len(df) == 5 * 4

    def test_respondent_ids_are_unique_and_sequential(self) -> None:
        df = make_dataset(
            n_respondents=4,
            n_items=6,
            items_per_task=3,
            repeats_per_item=2,
            true_utilities=np.zeros(6),
            seed=0,
        )
        ids = df["Response ID"].unique()
        assert len(ids) == 4
        # Default labels: R001, R002, R003, R004
        assert list(ids) == ["R001", "R002", "R003", "R004"]


# ----------------------------------------------------------------------
# Design balance and well-formedness
# ----------------------------------------------------------------------


class TestDesignBalance:
    @pytest.mark.parametrize(
        ("n_items", "items_per_task", "repeats"),
        [
            (6, 3, 2),  # 4 tasks
            (8, 4, 2),  # 4 tasks
            (10, 5, 3),  # 6 tasks
            (12, 4, 3),  # 9 tasks
        ],
    )
    def test_each_item_appears_repeats_times_per_respondent(
        self, n_items: int, items_per_task: int, repeats: int
    ) -> None:
        df = make_dataset(
            n_respondents=10,
            n_items=n_items,
            items_per_task=items_per_task,
            repeats_per_item=repeats,
            true_utilities=np.zeros(n_items),
            seed=0,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        for resp_id, resp_df in df.groupby("Response ID"):
            counts = pd.Series(resp_df[attr_cols].values.ravel()).value_counts()
            # Every item must appear exactly `repeats` times for this respondent.
            assert (counts == repeats).all(), (
                f"Respondent {resp_id} has unbalanced displays: {counts.to_dict()}"
            )

    def test_no_item_appears_twice_within_same_task(self) -> None:
        df = make_dataset(
            n_respondents=8,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.zeros(8),
            seed=7,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        for _, row in df.iterrows():
            displayed = row[attr_cols].tolist()
            assert len(set(displayed)) == len(displayed), f"Duplicate item within task: {displayed}"


class TestWellFormedness:
    def test_most_is_always_among_displayed_items(self) -> None:
        df = make_dataset(
            n_respondents=20,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
            seed=42,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        for _, row in df.iterrows():
            assert row["Most"] in row[attr_cols].tolist()

    def test_least_is_always_among_displayed_items(self) -> None:
        df = make_dataset(
            n_respondents=20,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
            seed=42,
        )
        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        for _, row in df.iterrows():
            assert row["Least"] in row[attr_cols].tolist()

    def test_most_and_least_are_different(self) -> None:
        df = make_dataset(
            n_respondents=20,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
            seed=42,
        )
        assert (df["Most"] != df["Least"]).all()


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_gives_identical_output(self) -> None:
        kwargs = dict(
            n_respondents=10,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.array([1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
            seed=123,
        )
        df_a = make_dataset(**kwargs)
        df_b = make_dataset(**kwargs)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_different_seeds_give_different_output(self) -> None:
        base_kwargs = dict(
            n_respondents=20,
            n_items=8,
            items_per_task=4,
            repeats_per_item=2,
            true_utilities=np.array([1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]),
        )
        df_a = make_dataset(seed=1, **base_kwargs)
        df_b = make_dataset(seed=2, **base_kwargs)
        # Vanishingly unlikely to be identical with these dimensions.
        assert not df_a.equals(df_b)


# ----------------------------------------------------------------------
# Utility -> choice probability
# ----------------------------------------------------------------------


class TestChoiceModel:
    def test_higher_utility_item_is_picked_as_most_more_often(self) -> None:
        # Item index 0 has utility 5 (much higher than all others).
        # When item 0 is displayed in a task, it should be picked as
        # Most far more often than a uniform 1/items_per_task baseline.
        n_items = 8
        items_per_task = 4
        true_u = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        df = make_dataset(
            n_respondents=200,
            n_items=n_items,
            items_per_task=items_per_task,
            repeats_per_item=3,
            true_utilities=true_u,
            seed=99,
        )

        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        # Tasks where the high-utility item was displayed.
        contains_item0 = df[attr_cols].apply(lambda r: "Item 1" in r.values, axis=1)
        tasks_with_item0 = df[contains_item0]
        assert len(tasks_with_item0) > 0
        pick_rate = (tasks_with_item0["Most"] == "Item 1").mean()
        baseline = 1.0 / items_per_task
        # With a 5-unit utility gap the share should be near 1.
        assert pick_rate > 0.85, (
            f"High-utility item picked as Most only {pick_rate:.1%} of the time "
            f"(baseline {baseline:.1%}); choice model may be broken."
        )

    def test_lowest_utility_item_is_picked_as_least_more_often(self) -> None:
        n_items = 8
        items_per_task = 4
        true_u = np.zeros(n_items)
        true_u[0] = -5.0  # Item 1 is very unattractive

        df = make_dataset(
            n_respondents=200,
            n_items=n_items,
            items_per_task=items_per_task,
            repeats_per_item=3,
            true_utilities=true_u,
            seed=99,
        )

        attr_cols = [c for c in df.columns if c.startswith("Attribute")]
        contains_item0 = df[attr_cols].apply(lambda r: "Item 1" in r.values, axis=1)
        tasks_with_item0 = df[contains_item0]
        pick_rate = (tasks_with_item0["Least"] == "Item 1").mean()
        assert pick_rate > 0.85


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


class TestValidation:
    def test_impossible_balanced_design_raises(self) -> None:
        # 7 * 2 = 14 displays per respondent; not divisible by 4.
        with pytest.raises(SyntheticDesignError, match="not divisible"):
            make_dataset(
                n_respondents=5,
                n_items=7,
                items_per_task=4,
                repeats_per_item=2,
                true_utilities=np.zeros(7),
                seed=0,
            )

    def test_items_per_task_larger_than_n_items_raises(self) -> None:
        with pytest.raises(SyntheticDesignError, match="items_per_task"):
            make_dataset(
                n_respondents=5,
                n_items=3,
                items_per_task=5,
                repeats_per_item=2,
                true_utilities=np.zeros(3),
                seed=0,
            )

    def test_utilities_length_mismatch_raises(self) -> None:
        with pytest.raises(SyntheticDesignError, match="true_utilities"):
            make_dataset(
                n_respondents=5,
                n_items=6,
                items_per_task=3,
                repeats_per_item=2,
                true_utilities=np.zeros(5),
                seed=0,
            )

    def test_negative_respondents_raises(self) -> None:
        with pytest.raises(SyntheticDesignError):
            make_dataset(
                n_respondents=0,
                n_items=6,
                items_per_task=3,
                repeats_per_item=2,
                true_utilities=np.zeros(6),
                seed=0,
            )

    def test_items_per_task_below_two_raises(self) -> None:
        with pytest.raises(SyntheticDesignError):
            make_dataset(
                n_respondents=5,
                n_items=6,
                items_per_task=1,
                repeats_per_item=2,
                true_utilities=np.zeros(6),
                seed=0,
            )
