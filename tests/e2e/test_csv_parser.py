"""Retroactive E2E tests for the Phase 0 CSV parser hardening.

Phase 0 replaced a fragile line-based CSV parser with one that handles
the full RFC 4180 cell content rules: doubled-double-quote escapes,
CRLF / LF / CR line endings, and embedded newlines within quoted
fields. The matching writer was updated to escape commas, quotes, and
newlines and to use CRLF line terminators on output.

These tests would have failed before Phase 0. They live in the e2e
layer because the parser is defined inline inside docs/analysis/
index.html and is most cheaply exercised by loading the page in a
real browser and calling the function via page.evaluate().

When the JS is extracted into a standalone module in Phase 3, these
tests will move down to a faster pure-JS test layer.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def _bring_page_up(page, analysis_url: str) -> None:
    """Navigate to the analysis page and wait until the in-page
    parseCSV / convertToCSV functions are defined.

    Pyodide loading is not required for these tests; we just need the
    page script to have run.
    """
    page.goto(analysis_url)
    page.wait_for_function("typeof parseCSV === 'function' && typeof convertToCSV === 'function'")


# ----------------------------------------------------------------------
# parseCSV
# ----------------------------------------------------------------------


class TestParseCSV:
    def test_handles_simple_csv(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = "Response ID,Attribute1,Most,Least\nR1,A,A,B\nR2,B,A,B\n"
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [
            {"Response ID": "R1", "Attribute1": "A", "Most": "A", "Least": "B"},
            {"Response ID": "R1" if False else "R2", "Attribute1": "B", "Most": "A", "Least": "B"},
        ]

    def test_handles_quoted_commas_inside_cell(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = 'Response ID,Label\nR1,"Hello, World"\n'
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [{"Response ID": "R1", "Label": "Hello, World"}]

    def test_handles_doubled_double_quote_escape(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = 'Response ID,Label\nR1,"Say ""hi"""\n'
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [{"Response ID": "R1", "Label": 'Say "hi"'}]

    def test_handles_crlf_line_endings(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = "Response ID,Attribute1\r\nR1,A\r\nR2,B\r\n"
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [
            {"Response ID": "R1", "Attribute1": "A"},
            {"Response ID": "R2", "Attribute1": "B"},
        ]

    def test_handles_lf_only_line_endings(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = "Response ID,Attribute1\nR1,A\nR2,B\n"
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [
            {"Response ID": "R1", "Attribute1": "A"},
            {"Response ID": "R2", "Attribute1": "B"},
        ]

    def test_handles_embedded_newline_inside_quoted_field(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = 'Response ID,Comment\nR1,"line one\nline two"\nR2,"single line"\n'
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert rows == [
            {"Response ID": "R1", "Comment": "line one\nline two"},
            {"Response ID": "R2", "Comment": "single line"},
        ]

    def test_skips_blank_lines(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = "Response ID,Attribute1\nR1,A\n\nR2,B\n"
        rows = page.evaluate("text => parseCSV(text)", csv)
        assert len(rows) == 2

    def test_raises_on_unclosed_quoted_field(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        csv = 'Response ID,Comment\nR1,"unclosed\n'
        # parseCSV throws; page.evaluate surfaces that as a PlaywrightError.
        with pytest.raises(Exception, match="unclosed"):
            page.evaluate("text => parseCSV(text)", csv)


# ----------------------------------------------------------------------
# convertToCSV (and round-trip with parseCSV)
# ----------------------------------------------------------------------


class TestConvertToCSV:
    def test_simple_round_trip(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        original = [
            {"Item": "A", "Score": 50},
            {"Item": "B", "Score": -25},
        ]
        csv = page.evaluate("data => convertToCSV(data)", original)
        round_tripped = page.evaluate("text => parseCSV(text)", csv)
        # CSV numbers come back as strings, so we compare normalized.
        assert round_tripped == [{"Item": "A", "Score": "50"}, {"Item": "B", "Score": "-25"}]

    def test_quotes_values_containing_comma(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        data = [{"Label": "Hello, World"}]
        csv = page.evaluate("d => convertToCSV(d)", data)
        # Header line, then the data line with the cell wrapped in quotes.
        assert '"Hello, World"' in csv

    def test_escapes_embedded_newline_and_round_trips(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        data = [{"Label": "line one\nline two"}, {"Label": "plain"}]
        csv = page.evaluate("d => convertToCSV(d)", data)
        # Embedded newline must be inside a quoted field.
        assert '"line one\nline two"' in csv
        # And must round-trip via parseCSV.
        round_tripped = page.evaluate("text => parseCSV(text)", csv)
        assert round_tripped == [
            {"Label": "line one\nline two"},
            {"Label": "plain"},
        ]

    def test_doubles_embedded_double_quotes(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        data = [{"Label": 'She said "hi"'}]
        csv = page.evaluate("d => convertToCSV(d)", data)
        # The cell must contain "" wherever the input had ".
        assert '"She said ""hi"""' in csv
        round_tripped = page.evaluate("text => parseCSV(text)", csv)
        assert round_tripped == [{"Label": 'She said "hi"'}]

    def test_uses_crlf_line_terminator(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        data = [{"X": "1"}, {"X": "2"}]
        csv = page.evaluate("d => convertToCSV(d)", data)
        # Per RFC 4180 our writer emits CRLF.
        assert csv.count("\r\n") >= 2

    def test_empty_data_yields_empty_string(self, page, analysis_url: str) -> None:
        _bring_page_up(page, analysis_url)
        result = page.evaluate("d => convertToCSV(d)", [])
        assert result == ""

    def test_round_trip_through_multiple_pathological_cells(self, page, analysis_url: str) -> None:
        """End-to-end check: data with commas, quotes, and newlines all
        round-trip cleanly."""
        _bring_page_up(page, analysis_url)
        data = [
            {"Item": "A, with comma", "Note": 'has "quote"'},
            {"Item": "B\nmultiline", "Note": "normal"},
            {"Item": "C", "Note": ""},
        ]
        csv = page.evaluate("d => convertToCSV(d)", data)
        round_tripped = page.evaluate("text => parseCSV(text)", csv)
        assert round_tripped == data
