"""File-reading helpers with encoding fallbacks.

CSV files from Qualtrics, SurveyMonkey, and other survey platforms
arrive in a handful of encodings (UTF-8 with or without BOM, UTF-16
from older Excel/Qualtrics exports, CP1252 from Windows Excel,
Latin-1 as a generic byte-tolerant fallback). The desktop GUI used
to call ``pd.read_csv(filename)`` directly, which defaults to UTF-8
and raises a cryptic ``UnicodeDecodeError`` on any non-UTF-8 file.

:func:`read_tabular_file` tries a sequence of encodings, succeeds on
the first that parses cleanly, and as a last resort reads with
``errors="replace"`` so the user at least gets *something* on
truly broken inputs rather than a crash.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

# Order matters: utf-8-sig must come before utf-8 to strip BOMs cleanly;
# utf-16 is checked next because Qualtrics is the most common source of
# UTF-16 CSVs we see; CP1252 catches Windows Excel; Latin-1 never fails
# at decode time so we keep it last (after the strict utf-8 attempt).
_ENCODING_FALLBACKS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "cp1252",
    "latin-1",
)


def read_tabular_file(
    path: str | Path,
    *,
    encodings: Iterable[str] = _ENCODING_FALLBACKS,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV or XLSX into a DataFrame, trying multiple encodings.

    Excel files (``.xlsx``) are read via :func:`pandas.read_excel`
    directly; the encoding fallback only applies to CSV.

    Parameters
    ----------
    path
        File system path. The extension determines whether the file
        is read as CSV or XLSX.
    encodings
        Encodings to try in order. The default covers the common
        survey-tool exports. Pass a smaller list to short-circuit.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Raises
    ------
    UnicodeDecodeError
        Only if every encoding in ``encodings`` failed AND the
        last-resort replacement read also failed (very unusual).
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if p.suffix.lower() == ".xlsx":
        return pd.read_excel(p)

    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(p, encoding=enc, **read_csv_kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue

    # Last resort: tolerate undecodable bytes so the user can still
    # see *something* and recover from a misencoded export.
    try:
        return pd.read_csv(p, encoding="utf-8", encoding_errors="replace", **read_csv_kwargs)
    except Exception as e:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"Could not decode {p} with any of {list(encodings)}"
        ) from (last_err or e)
