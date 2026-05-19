"""Input-validation tests for ``HierarchicalBayesMaxDiff._prepare_data``.

Phase 6 replaced the silent phantom-item-0 padding with explicit
ValueErrors. These tests pin the new behavior: any task whose
attribute cells, Most, or Least value falls outside the known item
set causes ``fit()`` to raise rather than producing a corrupted
analysis.

Tests are gated on the optional numpyro/jax dependencies.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tests.helpers import legacy_loader

_HAS_NUMPYRO = (
    importlib.util.find_spec("numpyro") is not None and importlib.util.find_spec("jax") is not None
)
pytestmark = [
    pytest.mark.hb,
    pytest.mark.skipif(not _HAS_NUMPYRO, reason="numpyro/jax not installed"),
]


def _good_dataset() -> pd.DataFrame:
    """A small balanced dataset that HB can fit without complaint."""
    return pd.DataFrame(
        [
            {
                "Response ID": "R1",
                "Attribute1": "A",
                "Attribute2": "B",
                "Attribute3": "C",
                "Most": "A",
                "Least": "C",
            },
            {
                "Response ID": "R1",
                "Attribute1": "A",
                "Attribute2": "B",
                "Attribute3": "D",
                "Most": "A",
                "Least": "D",
            },
            {
                "Response ID": "R2",
                "Attribute1": "A",
                "Attribute2": "B",
                "Attribute3": "C",
                "Most": "B",
                "Least": "C",
            },
            {
                "Response ID": "R2",
                "Attribute1": "B",
                "Attribute2": "C",
                "Attribute3": "D",
                "Most": "B",
                "Least": "D",
            },
        ]
    )


def test_prepare_data_accepts_well_formed_input() -> None:
    """Sanity check: the validation should not reject good data."""
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    # Just exercise _prepare_data; we don't need to actually fit.
    model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")
    assert model.n_items == 4
    assert model.n_respondents == 2


def test_prepare_data_raises_on_missing_attribute_cell() -> None:
    """A NaN in an attribute column means fewer recognized items than
    expected. Previously HB silently substituted item index 0; now it
    raises with a useful message."""
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    df.loc[0, "Attribute2"] = np.nan
    with pytest.raises(ValueError, match="attribute cells contained a known item"):
        model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")


def test_prepare_data_raises_on_unknown_most() -> None:
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    df.loc[0, "Most"] = "Z"  # Not displayed and not in the item set.
    with pytest.raises(ValueError, match="'Most' value 'Z' is not a known item"):
        model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")


def test_prepare_data_raises_on_unknown_least() -> None:
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    df.loc[0, "Least"] = "Z"
    with pytest.raises(ValueError, match="'Least' value 'Z' is not a known item"):
        model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")


def test_prepare_data_raises_when_most_not_in_displayed_set() -> None:
    """Most is a known item, but wasn't displayed in this task."""
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    # Row 0 displays A, B, C - Most is "D" which is a known item but
    # not in this task.
    df.loc[0, "Most"] = "D"
    with pytest.raises(ValueError, match="'Most' item 'D' not among the displayed"):
        model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")


def test_prepare_data_raises_when_least_not_in_displayed_set() -> None:
    legacy = legacy_loader.load_analyzer()
    model = legacy.HierarchicalBayesMaxDiff(n_iterations=10, n_warmup=10, n_chains=1)
    df = _good_dataset()
    df.loc[0, "Least"] = "D"
    with pytest.raises(ValueError, match="'Least' item 'D' not among the displayed"):
        model._prepare_data(df, ["Attribute1", "Attribute2", "Attribute3"], "Most", "Least")
