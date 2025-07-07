# src/cryotherm/stage.py
from __future__ import annotations

from typing import Callable

from scipy.optimize import brentq

from cryotherm.fridge import curve as get_fridge_curve


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
        self._load = external_load

        # updated each solver iteration
        self.net_heat_flow = 0.0

        # fridge cooling power function
        if isinstance(fridge_curve, str):
            fridge_curve = get_fridge_curve(fridge_curve)
        self._fridge = fridge_curve  # None or callable

    @property
    def has_fridge(self) -> bool:
        return self._fridge is not None

    @property
    def fridge_bounds(self):
        "Return (Tmin, Tmax) if the stage has a fridge, else (1, 500)."
        if not self.has_fridge:
            return 1.0, 500.0
        return self._fridge.T_min, self._fridge.T_max

    def target_temp_for_load(self, load_W: float) -> float:
        """
        Return the temperature at which the fridge delivers `load_W`.

        • If the stage has no fridge we simply return self.temperature
          (the caller will make residual = net_Q).
        • If the requested load exceeds the fridge envelope we clamp to
          the warm end (max lift formally = 0).
        """
        if not self.has_fridge:
            return self.temperature

        Tmin, Tmax = self.fridge_bounds
        f = lambda T: self.cooling_power(T) - load_W

        # Detect if load outside the lift curve
        if f(Tmin) < 0 and f(Tmax) < 0:
            return Tmin  # fridge can't remove that much → cold limit
        if f(Tmin) > 0 and f(Tmax) > 0:
            return Tmax  # load too small → warm end (≈0 W lift)

        return brentq(f, Tmin, Tmax, xtol=1e-4)

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
