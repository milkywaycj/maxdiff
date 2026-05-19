"""Color input parsing for chart styling.

The desktop GUI lets the user type a color into a text input. This
helper accepts the common forms (CSS4 named color, ``#RRGGBB`` /
``#RGB`` hex, or 6-char hex without the leading ``#``) and returns
a canonical matplotlib-compatible string, or ``None`` for
unrecognized input.

Phase 6 fix: previously any string beginning with ``#`` was returned
unchanged, so ``#GGGGGG`` or ``#zzz`` were accepted as "valid"
even though they could not be rendered. Hex strings are now
validated against the expected character set and length.
"""

from __future__ import annotations

import re

import matplotlib.colors as mcolors

# Accept the three RFC-ish hex forms: #RGB, #RRGGBB, #RRGGBBAA.
_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def process_color_input(color: object) -> str | None:
    """Normalize a user-supplied color string.

    Returns the canonical color string, or ``None`` if the input
    cannot be parsed.
    """
    if not color:
        return None
    color_str = str(color).strip()
    if not color_str:
        return None
    if color_str.lower() in mcolors.CSS4_COLORS:
        return color_str.lower()
    if color_str.startswith("#"):
        # Hash-prefixed string: must match a recognized hex pattern.
        return color_str if _HEX_RE.match(color_str) else None
    if len(color_str) == 6 and all(c in "0123456789ABCDEFabcdef" for c in color_str):
        return f"#{color_str}"
    try:
        return mcolors.to_hex(color_str)
    except ValueError:
        return None
