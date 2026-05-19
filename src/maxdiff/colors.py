"""Color input parsing for chart styling.

The desktop GUI lets the user type a color into a text input. This
helper accepts the common forms (CSS4 named color, ``#RRGGBB`` hex,
or 6-char hex without the leading ``#``) and returns a canonical
matplotlib-compatible string, or ``None`` for unrecognized input.

Known bug pinned by an xfail test in tests/unit/test_helpers.py: any
string beginning with ``#`` is returned unchanged without validating
the trailing characters are valid hex. Fix scheduled for Phase 6.
"""

from __future__ import annotations

import matplotlib.colors as mcolors


def process_color_input(color: object) -> str | None:
    """Normalize a user-supplied color string.

    Returns the canonical color string, or ``None`` if the input
    cannot be parsed.
    """
    if not color:
        return None
    color_str = str(color).strip()
    if color_str.lower() in mcolors.CSS4_COLORS:
        return color_str.lower()
    if color_str.startswith("#"):
        return color_str
    if len(color_str) == 6 and all(c in "0123456789ABCDEFabcdef" for c in color_str):
        return f"#{color_str}"
    try:
        return mcolors.to_hex(color_str)
    except ValueError:
        return None
