"""Golden-test fixtures and helpers.

Provides:

* ``update_goldens``  - CLI flag ``--update-goldens``. When set, tests
                        write their actual output to ``expected/`` and
                        pass; useful after an intentional change.

* ``assert_matches_golden`` - compare a DataFrame against a pinned CSV
                              under the expected/ directory. Supports
                              numerical tolerance for floats and is
                              deterministic in row ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

_EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Overwrite golden CSVs with the current test outputs.",
    )


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-goldens"))


@pytest.fixture
def assert_matches_golden(update_goldens: bool):
    """Return a comparator that asserts a DataFrame matches a pinned CSV.

    Usage::

        def test_my_pipeline(assert_matches_golden):
            result = run_pipeline(...)
            assert_matches_golden(result, "my_pipeline_result.csv")
    """

    def _compare(
        actual: pd.DataFrame,
        golden_filename: str,
        *,
        sort_by: list[str] | None = None,
        float_tol: float = 1e-9,
    ) -> None:
        path = _EXPECTED_DIR / golden_filename
        actual_sorted = actual.sort_values(sort_by).reset_index(drop=True) if sort_by else actual

        if update_goldens or not path.exists() or os.environ.get("UPDATE_GOLDENS"):
            path.parent.mkdir(parents=True, exist_ok=True)
            actual_sorted.to_csv(path, index=False, lineterminator="\n")
            if not update_goldens and not os.environ.get("UPDATE_GOLDENS"):
                pytest.fail(
                    f"Golden missing: {path.name}. Wrote current output; rerun the suite "
                    f"to verify, or pass --update-goldens to acknowledge."
                )
            return

        expected = pd.read_csv(path)
        if sort_by:
            expected = expected.sort_values(sort_by).reset_index(drop=True)

        # Column set / order match.
        assert list(actual_sorted.columns) == list(expected.columns), (
            f"Column mismatch in {golden_filename}:\n"
            f"  actual:   {list(actual_sorted.columns)}\n"
            f"  expected: {list(expected.columns)}"
        )
        # Row count match.
        assert len(actual_sorted) == len(expected), (
            f"Row count mismatch in {golden_filename}: "
            f"actual={len(actual_sorted)}, expected={len(expected)}"
        )
        # Cell-by-cell comparison with float tolerance.
        for col in actual_sorted.columns:
            actual_col = actual_sorted[col]
            expected_col = expected[col]
            if pd.api.types.is_numeric_dtype(actual_col) and pd.api.types.is_numeric_dtype(
                expected_col
            ):
                # numpy.allclose handles both float and integer dtypes.
                import numpy as np

                if not np.allclose(
                    actual_col.to_numpy(),
                    expected_col.to_numpy(),
                    atol=float_tol,
                    rtol=float_tol,
                    equal_nan=True,
                ):
                    diffs = actual_col.to_numpy() - expected_col.to_numpy()
                    raise AssertionError(
                        f"Numeric column '{col}' differs in {golden_filename}:\n"
                        f"  max abs diff = {abs(diffs).max():.6g}\n"
                        f"  actual:   {actual_col.tolist()}\n"
                        f"  expected: {expected_col.tolist()}"
                    )
            else:
                if not (actual_col.astype(str).tolist() == expected_col.astype(str).tolist()):
                    raise AssertionError(
                        f"Text column '{col}' differs in {golden_filename}:\n"
                        f"  actual:   {actual_col.tolist()}\n"
                        f"  expected: {expected_col.tolist()}"
                    )

    return _compare
