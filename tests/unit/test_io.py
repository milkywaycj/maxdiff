"""Tests for the encoding-fallback file reader.

Covers the common encodings we see from survey-platform exports
(UTF-8 with and without BOM, UTF-16, CP1252, Latin-1). The reader
must round-trip non-ASCII content cleanly without raising.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from maxdiff.io import read_tabular_file

CSV_CONTENT = (
    "Response ID,Attribute1,Attribute2,Attribute3,Most,Least\n"
    "R1,Café,Naïve,Façade,Café,Façade\n"
    "R2,Naïve,Café,Façade,Naïve,Façade\n"
)


def _write(path: Path, encoding: str, content: str = CSV_CONTENT) -> Path:
    if encoding == "utf-8-sig":
        # utf-8-sig is BOM + utf-8 bytes
        path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    else:
        path.write_bytes(content.encode(encoding))
    return path


@pytest.mark.parametrize(
    "encoding",
    ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"],
)
def test_round_trips_common_encodings(tmp_path: Path, encoding: str) -> None:
    csv_path = _write(tmp_path / f"data_{encoding}.csv", encoding)
    df = read_tabular_file(csv_path)
    assert list(df.columns) == [
        "Response ID",
        "Attribute1",
        "Attribute2",
        "Attribute3",
        "Most",
        "Least",
    ]
    assert df.iloc[0]["Attribute1"] == "Café"
    assert df.iloc[0]["Attribute3"] == "Façade"


def test_excel_file_routed_to_read_excel(tmp_path: Path) -> None:
    """XLSX files go through pandas.read_excel; the encoding fallback
    list is not consulted for them."""
    xlsx_path = tmp_path / "data.xlsx"
    df_orig = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    df_orig.to_excel(xlsx_path, index=False)
    df = read_tabular_file(xlsx_path)
    pd.testing.assert_frame_equal(df, df_orig)


def test_falls_back_to_replacement_on_garbage_bytes(tmp_path: Path) -> None:
    """A file with bytes that cannot be cleanly decoded by any
    enumerated encoding still produces a DataFrame (with replacement
    characters) rather than raising."""
    p = tmp_path / "broken.csv"
    # Bytes that are valid UTF-8 framework (the header and a row) plus
    # a stray byte sequence that is invalid UTF-8 / UTF-16 but is
    # decodable as Latin-1 (Latin-1 accepts any byte).
    p.write_bytes(b"A,B\n1,2\n\x81\x9d\n")
    df = read_tabular_file(p)
    assert "A" in df.columns
    assert len(df) == 2


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_tabular_file(tmp_path / "does_not_exist.csv")


def test_accepts_pathlib_and_string(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "data.csv", "utf-8")
    df_from_path = read_tabular_file(csv_path)
    df_from_str = read_tabular_file(str(csv_path))
    pd.testing.assert_frame_equal(df_from_path, df_from_str)
