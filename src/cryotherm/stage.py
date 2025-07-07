# src/cryotherm/stage.py
from __future__ import annotations

from typing import Callable


class Stage:
    """
    One thermal node (shell, component, etc.).

    Parameters
    ----------
    name : str
    T0   : float   initial (or fixed) temperature in K
    fridge_curve : Callable[[float], float] | None
        Cooling power [W] available *at* temperature T (positive value).
    fixed : bool
        If True, temperature is held at `T0` (not solved).
    external_load : float | Callable[[float], float]
        Extra heat load always applied (+W). Use a function for T-dependent load.
    """

    def __init__(
        self,
        name: str,
        T0: float,
        *,
        fridge_curve: Callable[[float], float] | None = None,
        fixed: bool = False,
        external_load: float | Callable[[float], float] = 0.0,
    ):
        self.name = name
        self.temperature = float(T0)
        self.fixed = bool(fixed)
        self._fridge = fridge_curve
        self._load = external_load

        # updated each solver iteration
        self.net_heat_flow = 0.0

    # ---------------------------------------
    def cooling_power(self, T: float) -> float:
        return 0.0 if self._fridge is None else float(self._fridge(T))

    def external_load(self, T: float) -> float:
        if callable(self._load):
            return float(self._load(T))
        return float(self._load)

    # ---------------------------------------
    def __repr__(self) -> str:
        state = "fixed" if self.fixed else "free"
        return f"<Stage {self.name}: {self.temperature:.3f} K ({state})>"
