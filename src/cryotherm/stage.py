# src/cryotherm/stage.py
from __future__ import annotations

from typing import Callable

import numpy as np
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
        target_temperature: float | None = None,
    ):
        self.name = name
        self.temperature = float(T0)
        self.fixed = bool(fixed)
        self._load = external_load
        self.target_temperature = (
            float(target_temperature) if target_temperature is not None else None
        )

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

    def target_temp_for_load(self, load_W: float, *, n_scan: int = 200) -> float:
        """
        Return T such that P_fridge(T) ≈ load_W.

        Works for curves where lift increases or decreases with T,
        and clamps to Tmin/Tmax if the requested load is unattainable.
        """
        if not self.has_fridge:
            return self.temperature  # plain stage: nothing to do

        Tmin, Tmax = self.fridge_bounds
        P = self.cooling_power

        # Early out if an endpoint already matches
        if np.isclose(P(Tmin), load_W, rtol=1e-4, atol=1e-4):
            return Tmin
        if np.isclose(P(Tmax), load_W, rtol=1e-4, atol=1e-4):
            return Tmax

        # Does load lie within [Pmin, Pmax] ?
        Pmin, Pmax = min(P(Tmin), P(Tmax)), max(P(Tmin), P(Tmax))
        if load_W < Pmin:
            return Tmin if P(Tmin) < P(Tmax) else Tmax  # pick colder end
        if load_W > Pmax:
            return Tmax if P(Tmax) > P(Tmin) else Tmin  # pick warmer end

        # Bracket a root by linear scan (robust to non-monotonic wiggles)
        Ts = np.linspace(Tmin, Tmax, n_scan)
        Ps = P(Ts) - load_W
        sign = np.sign(Ps)

        # find first sign change
        idx = np.where(np.diff(sign))[0]
        if idx.size == 0:  # numerical corner-case
            return Tmin

        i = idx[0]
        T_low, T_high = Ts[i], Ts[i + 1]
        return brentq(lambda T: P(T) - load_W, T_low, T_high, xtol=1e-4)

    # ---------------------------------------
    def cooling_power(self, T):
        """
        Return cooler lift at temperature(s) T.

        * Works for scalar or NumPy array input.
        * If no fridge attached → 0.
        """
        if self._fridge is None:
            return 0.0

        # vector-friendly: if T is ndarray leave it be, else cast to float
        if isinstance(T, np.ndarray):
            return self._fridge(T)  # already vectorised
        else:
            return float(self._fridge(float(T)))

    def external_load(self, T: float) -> float:
        if callable(self._load):
            return float(self._load(T))
        return float(self._load)

    # ---------------------------------------
    def __repr__(self) -> str:
        state = "fixed" if self.fixed else "free"
        return f"<Stage {self.name}: {self.temperature:.3f} K ({state})>"
