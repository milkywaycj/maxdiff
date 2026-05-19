# Golden output tests

This directory contains pinned expected outputs ("goldens") for the
analysis pipeline. Each test:

1. Runs a deterministic synthetic dataset through one of the legacy
   functions (`calculate_observed_percentages`, `bootstrap_analysis`,
   `calculate_display_statistics`, etc.).
2. Asserts the output matches a checked-in CSV byte-for-byte (or
   value-for-value, with tolerance for floats).

## Purpose

Goldens are the safety net for invasive refactors. When the legacy
analyzer is extracted into the `maxdiff_core` package (Phase 3), or
when numerical fixes intentionally change outputs (Phase 4), the
golden diffs surface every change explicitly — there are no silent
shifts.

## Updating goldens

When an output is *expected* to change (e.g., a Phase 4 fix that
correctly recomputes a score), regenerate the affected goldens with:

    pytest tests/golden/ --update-goldens

Then review the diff in `git status`. If the diff matches the
intended change, commit it with a message that quotes the
corresponding test name and explains the intentional difference.

If you ever see a golden diff *without* an accompanying intentional
change, you have a regression. Investigate before regenerating.

## File layout

* `test_*.py` - test files that compare against goldens
* `expected/` - the pinned CSV outputs
* `conftest.py` - shared helpers for golden comparison
