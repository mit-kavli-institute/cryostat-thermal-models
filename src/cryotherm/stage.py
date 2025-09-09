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
    T0   : float
    fridge_curve : Callable[[float], float] | str | None
        Cooling power [W] available *at* temperature T (positive).
    fixed : bool
        If True, temperature is held at `T0`.
    external_load : float | Callable[[float], float]
        Baseline (parasitic) heat load (+W). May be T-dependent.
        NOTE: Trim-heater power is solved separately when `target_temperature`
        is provided; you do NOT include it here.
    target_temperature : float | None
        If provided, this stage’s temperature is clamped to this value and the
        solver will size a non-negative heater power to satisfy energy balance.
    """

    def __init__(
        self,
        name: str,
        T0: float,
        *,
        fridge_curve: Callable[[float], float] | str | None = None,
        fixed: bool = False,
        external_load: float | Callable[[float], float] = 0.0,
        target_temperature: float | None = None,
    ):
        self.name = name
        self.temperature = float(T0)
        self.fixed = bool(fixed)

        # --- NEW: split baseline vs. heater ------------------------------
        self._base_load = external_load  # baseline electronics etc.
        self._heater_W: float = 0.0  # sized by solver if target_temperature

        self.target_temperature = (
            float(target_temperature) if target_temperature is not None else None
        )

        # updated each solver iteration (for reporting/visualization)
        self.net_heat_flow = 0.0

        # fridge cooling power function
        if isinstance(fridge_curve, str):
            fridge_curve = get_fridge_curve(fridge_curve)
        self._fridge = fridge_curve  # None or callable

    # ------------------------ properties --------------------------------
    @property
    def has_fridge(self) -> bool:
        return self._fridge is not None

    @property
    def fridge_bounds(self):
        "Return (Tmin, Tmax) if the stage has a fridge, else (1, 500)."
        if not self.has_fridge:
            return 1.0, 500.0
        return self._fridge.T_min, self._fridge.T_max

    @property
    def heater_power(self) -> float:
        """Current trim-heater power (W)."""
        return float(self._heater_W)

    # ------------------------ fridge helpers ----------------------------
    def target_temp_for_load(self, load_W: float, *, n_scan: int = 200) -> float:
        """Return T such that P_fridge(T) ≈ load_W. Clamps to bounds if needed."""
        if not self.has_fridge:
            return self.temperature

        Tmin, Tmax = self.fridge_bounds
        P = self.cooling_power

        if np.isclose(P(Tmin), load_W, rtol=1e-4, atol=1e-4):
            return Tmin
        if np.isclose(P(Tmax), load_W, rtol=1e-4, atol=1e-4):
            return Tmax

        Pmin, Pmax = min(P(Tmin), P(Tmax)), max(P(Tmin), P(Tmax))
        if load_W < Pmin:
            return Tmin if P(Tmin) < P(Tmax) else Tmax
        if load_W > Pmax:
            return Tmax if P(Tmax) > P(Tmin) else Tmin

        Ts = np.linspace(Tmin, Tmax, n_scan)
        Ps = P(Ts) - load_W
        sign = np.sign(Ps)
        idx = np.where(np.diff(sign))[0]
        if idx.size == 0:  # numerical corner-case
            return Tmin
        i = idx[0]
        T_low, T_high = Ts[i], Ts[i + 1]
        return brentq(lambda T: P(T) - load_W, T_low, T_high, xtol=1e-4)

    def cooling_power(self, T):
        """Cooler lift at temperature(s) T (0 if no fridge)."""
        if self._fridge is None:
            return 0.0
        if isinstance(T, np.ndarray):
            return self._fridge(T)
        return float(self._fridge(float(T)))

    # ------------------------ loads -------------------------------------
    def _baseline_load(self, T: float) -> float:
        """Baseline parasitic load only (W)."""
        if callable(self._base_load):
            return float(self._base_load(T))
        return float(self._base_load)

    def external_load(self, T: float) -> float:
        """Total inbound load (baseline + heater)."""
        return self._baseline_load(T) + self._heater_W

    # ------------------------ misc --------------------------------------
    def __repr__(self) -> str:
        state = "fixed" if self.fixed else "free"
        return f"<Stage {self.name}: {self.temperature:.3f} K ({state})>"
