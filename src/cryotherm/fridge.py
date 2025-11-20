from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np

from cryotherm import DATA_PATH

# Directory that ships with the package
_FRIDGE_DIR = DATA_PATH / "fridges"

# internal cache avoids re-reading CSVs
_FRIDGE_CACHE: Dict[str, Callable[[float | np.ndarray], float]] = {}

# index: lower-case name -> Path
_FRIDGE_INDEX: Dict[str, Path] = {p.stem.lower(): p for p in _FRIDGE_DIR.glob("*.csv")}


def available() -> list[str]:
    """Return the list of fridge names shipped with the package."""
    # If you want to preserve on-disk casing in this listing:
    return sorted(p.stem for p in _FRIDGE_INDEX.values())
    # Or, if you prefer fully lowercased names, use:
    # return sorted(_FRIDGE_INDEX.keys())


def curve(name: str) -> Callable[[float | np.ndarray], float]:
    """
    Return **cooling-power function**  P(T)  for the given fridge *name*.

    Example
    -------
    >>> CryoTel = fridge.curve("CryotelGT")
    >>> CryoTel(45)      # W
    0.37
    """
    key = name.strip().lower()
    if key in _FRIDGE_CACHE:
        return _FRIDGE_CACHE[key]

    csv_path = _FRIDGE_INDEX.get(key)
    if csv_path is None:
        raise ValueError(
            f"No fridge named '{name}'.  Available: {', '.join(available())}"
        )

    T, P = np.loadtxt(csv_path, delimiter=",", unpack=True)

    Tmin, Tmax = T.min(), T.max()

    def _P(Tquery):
        Tq = np.asanyarray(Tquery, dtype=float)
        lift = np.interp(Tq, T, P, left=0.0, right=0.0)
        return lift if isinstance(Tquery, np.ndarray) else float(lift)

    _P.T_min, _P.T_max = float(Tmin), float(Tmax)
    _FRIDGE_CACHE[key] = _P
    return _P
