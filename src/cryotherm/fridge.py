"""
cryotherm.fridge
~~~~~~~~~~~~~~~~
Convenience layer that turns fridge‐load curves stored under
`DATA_PATH / "fridges"` into callables you can hand straight to
`Stage(fridge_curve=…)`.

All CSVs must be 2-column:  Temperature [K] ,  Lift [W]
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Callable, Dict

import numpy as np

# ---------------------------------------------------------------------
# internal cache avoids re-reading CSVs
_FRIDGE_CACHE: Dict[str, Callable[[float | np.ndarray], float]] = {}

# directory that ships with the package
from cryotherm import DATA_PATH

_FRIDGE_DIR = DATA_PATH / "fridges"


def available() -> list[str]:
    """Return the list of fridge names shipped with the package."""
    return sorted(p.stem for p in _FRIDGE_DIR.glob("*.csv"))


def curve(name: str) -> Callable[[float | np.ndarray], float]:
    """
    Return **cooling-power function**  P(T)  for the given fridge *name*.

    Example
    -------
    >>> CryoTel = fridge.curve("CryotelGT")
    >>> CryoTel(45)      # W
    0.37
    """
    name = name.strip().lower()
    if name in _FRIDGE_CACHE:
        return _FRIDGE_CACHE[name]

    csv_path = _FRIDGE_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise ValueError(
            f"No fridge named '{name}'.  " f"Available: {', '.join(available())}"
        )

    T, P = np.loadtxt(csv_path, delimiter=",", unpack=True)

    # build a vectorised interpolator with zero power outside the tabulated range
    Tmin, Tmax = T.min(), T.max()

    def _P(Tquery):
        Tq = np.asanyarray(Tquery, dtype=float)
        lift = np.interp(Tq, T, P, left=0.0, right=0.0)
        return lift if isinstance(Tquery, np.ndarray) else float(lift)

    _P.T_min, _P.T_max = float(Tmin), float(Tmax)
    _FRIDGE_CACHE[name] = _P
    return _P
