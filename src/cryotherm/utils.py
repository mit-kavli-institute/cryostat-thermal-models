"""
cryotherm.utils
Shared geometry helpers.
"""

from __future__ import annotations

import math
from typing import Any, Literal

π = math.pi


# ----------------------------------------------------------------------
# Cross-sectional area  (for conductive straps / wires / tubes)
# ----------------------------------------------------------------------
def cs_area(
    kind: Literal["rect", "cylinder", "tube"],
    **kw: Any,
) -> float:
    """
    Return **cross-sectional area** [m²].

    Parameters
    ----------
    kind : "rect" | "cylinder" | "tube"
    kw   : dimensions (see below)

    Accepted keywords
    -----------------
    rect      : width, thickness
    cylinder  : outer_dia
    tube      : outer_dia, inner_dia
    """
    if kind == "rect":
        return float(kw["width"]) * float(kw["thickness"])

    if kind == "cylinder":
        d = float(kw["outer_dia"])
        return π * d**2 / 4.0

    if kind == "tube":
        do = float(kw["outer_dia"])
        di = float(kw["inner_dia"])
        return π / 4.0 * (do**2 - di**2)

    raise ValueError(f"Unknown cross-section kind '{kind}'")


# ----------------------------------------------------------------------
# Surface area (for radiation links)
# ----------------------------------------------------------------------
def surf_area(
    shape: Literal["plate", "cylinder", "box"],
    **kw: Any,
) -> float:
    """
    Return **surface area** [m²] for simple shapes.

    plate      : diameter  |  width, length
    cylinder   : diameter, height   (lateral area only)
    box        : width, length, height
    """
    if shape == "plate":
        if "diameter" in kw:
            d = float(kw["diameter"])
            return π * (d / 2.0) ** 2
        return float(kw["width"]) * float(kw["length"])

    if shape == "cylinder":
        d = float(kw["diameter"])
        h = float(kw["height"])
        return π * d * h

    if shape == "box":
        w, l, h = map(float, (kw["width"], kw["length"], kw["height"]))  # noqa: E741
        return 2.0 * (w * l + w * h + l * h)

    raise ValueError(f"Unknown surface shape '{shape}'")
