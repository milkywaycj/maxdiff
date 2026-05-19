"""Synthetic MaxDiff data generator.

Produces DataFrames in the exact format the analyzer expects::

    Response ID | Attribute1 | Attribute2 | ... | Most | Least

The generator does two things:

1. Builds a *balanced design* per respondent: across that respondent's
   tasks, every item appears exactly ``repeats_per_item`` times, and
   no item appears twice within a single task.

2. Simulates best/worst *choices* under the standard MaxDiff /
   Bradley-Terry-Luce model. Given a true utility vector ``u``, for
   each task we add independent Gumbel(0,1) shocks to the utilities
   of the displayed items and pick:

   * Most = argmax over (u + epsilon_best)
   * Least = argmin over (u + epsilon_worst), constrained to be
     different from Most

This is the assumption under which Hierarchical Bayes MaxDiff is
identifiable and under which count-analysis recovers the utility
ordering. Tests in the statistical layer use this generator with a
known ``true_utilities`` to verify the analyzer recovers it.

Design construction uses a shuffle-and-retry strategy similar to the
browser design tool: shuffle a multiset pool of items, slice into
tasks, and if any task has a duplicate, retry. For pathological
configurations the retry succeeds quickly in practice; we cap retries
to surface impossible designs as ``SyntheticDesignError`` rather than
hanging.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

_MAX_DESIGN_RETRIES: Final[int] = 200


class SyntheticDesignError(ValueError):
    """Raised for any infeasible or malformed generator input."""


def make_dataset(
    n_respondents: int,
    n_items: int,
    items_per_task: int,
    repeats_per_item: int,
    true_utilities: np.ndarray | Sequence[float],
    seed: int | None = None,
    item_labels: Sequence[str] | None = None,
    respondent_id_format: str = "R{:03d}",
) -> pd.DataFrame:
    """Generate a synthetic MaxDiff dataset.

    Parameters
    ----------
    n_respondents
        Number of respondents. Must be positive.
    n_items
        Number of distinct items in the study. Must be >= 2.
    items_per_task
        How many items are shown together in each choice task. Must be
        in the range [2, n_items].
    repeats_per_item
        How many times each item appears in each respondent's task set.
        Must be >= 1.
    true_utilities
        Length-``n_items`` array of true utilities. Higher values are
        more likely to be picked as Most; lower values as Least. Only
        the *differences* between utilities matter for choice
        probabilities; an additive constant has no effect.
    seed
        Optional integer seed for reproducibility. ``None`` produces a
        fresh random stream.
    item_labels
        Optional length-``n_items`` sequence of labels. Defaults to
        "Item 1", "Item 2", etc.
    respondent_id_format
        Format string used to generate Response IDs from sequential
        integers starting at 1. Default produces R001, R002, ...

    Returns
    -------
    pd.DataFrame
        One row per (respondent, task) with columns
        ``Response ID, Attribute1..Attribute<items_per_task>, Most, Least``.
        Most and Least are item labels (strings), guaranteed to be
        among the items displayed in that task and guaranteed
        different from each other.

    Raises
    ------
    SyntheticDesignError
        On any infeasible input: non-positive sizes, items_per_task
        outside [2, n_items], a (n_items * repeats_per_item) that is
        not divisible by items_per_task, or a utility vector of the
        wrong length.
    """
    _validate(n_respondents, n_items, items_per_task, repeats_per_item, true_utilities)

    utilities = np.asarray(true_utilities, dtype=np.float64)
    if item_labels is None:
        item_labels = [f"Item {i + 1}" for i in range(n_items)]
    else:
        item_labels = list(item_labels)
        if len(item_labels) != n_items:
            raise SyntheticDesignError(
                f"item_labels has length {len(item_labels)} but n_items is {n_items}"
            )

    rng = np.random.default_rng(seed)

    attr_columns = [f"Attribute{i + 1}" for i in range(items_per_task)]

    rows: list[dict[str, object]] = []
    for r in range(n_respondents):
        respondent_id = respondent_id_format.format(r + 1)
        design = _build_balanced_design(n_items, items_per_task, repeats_per_item, rng)

        for task_items in design:
            most_idx, least_idx = _sample_best_worst(task_items, utilities, rng)
            row: dict[str, object] = {"Response ID": respondent_id}
            for i, item_idx in enumerate(task_items):
                row[attr_columns[i]] = item_labels[item_idx]
            row["Most"] = item_labels[most_idx]
            row["Least"] = item_labels[least_idx]
            rows.append(row)

    columns = ["Response ID", *attr_columns, "Most", "Least"]
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _validate(
    n_respondents: int,
    n_items: int,
    items_per_task: int,
    repeats_per_item: int,
    true_utilities: np.ndarray | Sequence[float],
) -> None:
    if n_respondents <= 0:
        raise SyntheticDesignError(f"n_respondents must be positive, got {n_respondents}")
    if n_items < 2:
        raise SyntheticDesignError(f"n_items must be at least 2, got {n_items}")
    if items_per_task < 2:
        raise SyntheticDesignError(f"items_per_task must be at least 2, got {items_per_task}")
    if items_per_task > n_items:
        raise SyntheticDesignError(
            f"items_per_task ({items_per_task}) cannot exceed n_items ({n_items})"
        )
    if repeats_per_item < 1:
        raise SyntheticDesignError(f"repeats_per_item must be at least 1, got {repeats_per_item}")
    total_displays = n_items * repeats_per_item
    if total_displays % items_per_task != 0:
        raise SyntheticDesignError(
            f"n_items * repeats_per_item ({total_displays}) is not divisible by "
            f"items_per_task ({items_per_task}). Choose values so the product divides cleanly."
        )
    utilities = np.asarray(true_utilities)
    if utilities.ndim != 1 or utilities.shape[0] != n_items:
        raise SyntheticDesignError(
            f"true_utilities must be a 1-D array of length n_items={n_items}, "
            f"got shape {utilities.shape}"
        )
    if not np.all(np.isfinite(utilities)):
        raise SyntheticDesignError("true_utilities must contain only finite values")


def _build_balanced_design(
    n_items: int,
    items_per_task: int,
    repeats_per_item: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Return a balanced design as a list of tasks, each a list of item indices.

    Two-stage algorithm:

    1. Fast path: build a multiset pool (each item repeated
       ``repeats_per_item`` times), shuffle, slice into tasks of size
       ``items_per_task``. Random shuffle finds a valid design quickly
       when items_per_task is comfortably smaller than n_items.

    2. Greedy fallback: when items_per_task is close to (or equal to)
       n_items, the constraint is tight and random shuffling rarely
       lands on a valid arrangement. We instead build tasks one slot
       at a time, picking from items that still have copies remaining
       and aren't already in the current task, preferring those with
       the most remaining copies (to avoid late starvation). This
       always succeeds for any design that passes input validation,
       at the cost of slightly less variety than a true uniform draw.
    """
    n_tasks = (n_items * repeats_per_item) // items_per_task
    pool_template = np.repeat(np.arange(n_items), repeats_per_item)

    for _ in range(_MAX_DESIGN_RETRIES):
        pool = pool_template.copy()
        rng.shuffle(pool)
        tasks = [
            pool[i * items_per_task : (i + 1) * items_per_task].tolist() for i in range(n_tasks)
        ]
        if all(len(set(task)) == items_per_task for task in tasks):
            return tasks

    # Greedy fallback - guaranteed to succeed for any feasible design.
    return _build_with_greedy_fill(n_items, items_per_task, repeats_per_item, n_tasks, rng)


def _build_with_greedy_fill(
    n_items: int,
    items_per_task: int,
    repeats_per_item: int,
    n_tasks: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Build a balanced design by greedy frequency-first selection.

    For each task slot in turn, pick an item that still has copies in
    the pool and isn't already in the current task, preferring items
    with the most remaining copies. Picking the most-frequent
    candidate prevents starvation where one item gets stuck and
    cannot fit into the remaining task slots.
    """
    remaining = np.full(n_items, repeats_per_item, dtype=np.int64)
    tasks: list[list[int]] = []
    for _task_idx in range(n_tasks):
        used: set[int] = set()
        task: list[int] = []
        for _slot in range(items_per_task):
            candidates = [i for i in range(n_items) if remaining[i] > 0 and i not in used]
            if not candidates:
                raise SyntheticDesignError(
                    "Greedy fill exhausted while building a design that should be "
                    f"feasible (n_items={n_items}, items_per_task={items_per_task}, "
                    f"repeats_per_item={repeats_per_item})."
                )
            max_remaining = max(remaining[i] for i in candidates)
            best = [i for i in candidates if remaining[i] == max_remaining]
            choice = int(rng.choice(best))
            task.append(choice)
            used.add(choice)
            remaining[choice] -= 1
        tasks.append(task)
    return tasks


def _sample_best_worst(
    task_items: list[int],
    utilities: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (best_idx, worst_idx) for a task under the Gumbel MaxDiff model.

    Returns the *original item indices* (not task-local positions). The
    two returned indices are guaranteed to be different.
    """
    u_task = utilities[task_items]
    # Independent Gumbel(0, 1) shocks for the best and worst draws.
    # The standard inverse-CDF trick: -log(-log(uniform)).
    gumbel_best = -np.log(-np.log(rng.uniform(size=u_task.shape)))
    gumbel_worst = -np.log(-np.log(rng.uniform(size=u_task.shape)))

    best_local = int(np.argmax(u_task + gumbel_best))

    # Worst is sampled from the remaining items (best != worst is a
    # MaxDiff invariant; respondents cannot pick the same option as
    # both most and least preferred).
    mask = np.ones(u_task.shape[0], dtype=bool)
    mask[best_local] = False
    remaining_scores = -(u_task + gumbel_worst)  # negate so argmax = argmin
    remaining_scores[~mask] = -np.inf
    worst_local = int(np.argmax(remaining_scores))

    return task_items[best_local], task_items[worst_local]
