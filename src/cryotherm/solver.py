# src/cryotherm/solver.py
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import least_squares


class ThermalModel:
    """
    Solve steady-state temperatures and, where requested, heater powers.
    """

    def __init__(
        self,
        stages: List,
        conductors: List = None,
        radiators: List = None,
    ):
        self.stages = stages
        self.conductors = conductors or []
        self.radiators = radiators or []

        # split free variables
        self.temp_vars = [
            s for s in stages if (not s.fixed) and s.target_temperature is None
        ]
        self.heat_vars = [s for s in stages if s.target_temperature is not None]

    # -----------------------------------------------------------------
    # residuals
    # -----------------------------------------------------------------
    def _residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Unknown vector ``x``  =  [temps…, heater_powers…]
        """
        N_t = len(self.temp_vars)

        # 0) assign trial values
        for i, s in enumerate(self.temp_vars):
            s.temperature = x[i]

        for j, s in enumerate(self.heat_vars, start=N_t):
            s._load = x[j]  # heater power (W)
            s.temperature = s.target_temperature  # fixed T each iteration

        # 1) zero inbound tallies
        for s in self.stages:
            s._q_in = s.external_load(s.temperature)  # includes heater if any

        # 2) conduction
        for c in self.conductors:
            q = c.heat_flow(c.stage1.temperature, c.stage2.temperature)
            c.stage1._q_in -= q
            c.stage2._q_in += q

        # 3) radiation
        for r in self.radiators:
            if r.stage2 is None:
                q = r.heat_flow(r.stage1.temperature, None)
                r.stage1._q_in -= q
            else:
                q = r.heat_flow(r.stage1.temperature, r.stage2.temperature)
                r.stage1._q_in -= q
                r.stage2._q_in += q

        # 4) build residuals
        res = []

        #   a) temperature unknowns
        for s in self.temp_vars:
            if s.has_fridge:
                T_target = s.target_temp_for_load(s._q_in)
                res.append(s.temperature - T_target)
            else:
                res.append(s._q_in)  # want 0 W net

        #   b) heater-power unknowns (temperature is clamped)
        for s in self.heat_vars:
            res.append(s._q_in)  # want 0 W net

        # diagnostics (optional)
        for s in self.stages:
            s.net_heat_flow = s._q_in

        return np.array(res, dtype=float)

    @staticmethod
    def _initial_heater_guess(stage):
        """Return a numeric starting value even if user passed a callable."""
        return 0.0 if callable(stage._load) else float(stage._load)

    # -----------------------------------------------------------------
    # public solve
    # -----------------------------------------------------------------
    def solve(self, tol: float = 1e-6):
        # build initial guess and bounds
        x0, lo, hi = [], [], []

        for s in self.temp_vars:
            Tmin, Tmax = s.fridge_bounds
            Tguess = np.clip(s.temperature, Tmin, Tmax)
            if s.has_fridge and not (Tmin <= s.temperature <= Tmax):
                Tguess = 0.5 * (Tmin + Tmax)
            x0.append(Tguess)
            lo.append(Tmin)
            hi.append(Tmax)

        for s in self.heat_vars:
            # heater power guess = previous external_load (≥0)
            P0 = self._initial_heater_guess(s)
            x0.append(P0)
            lo.append(0.0)  # heater cannot cool
            hi.append(1e6)  # 1 MW upper cap

        sol = least_squares(
            self._residuals,
            np.array(x0),
            bounds=(np.array(lo), np.array(hi)),
            xtol=tol,
            ftol=tol,
            gtol=tol,
        )
        if not sol.success:
            raise RuntimeError("Solver failed: " + sol.message)

        # write back final values (temps already set inside residuals loop)
        N_t = len(self.temp_vars)
        for j, s in enumerate(self.heat_vars, start=N_t):
            s._load = sol.x[j]  # final heater power (W)

        return sol

    def report(self):
        print("\n=== Balance at current temperatures ===")
        for s in self.stages:
            print(f"{s.name:15s}: T={s.temperature:7.2f} K")
        print("\nConduction:")
        for c in self.conductors:
            # always report hot→cold
            sH, sC = (
                (c.stage1, c.stage2)
                if c.stage1.temperature > c.stage2.temperature
                else (c.stage2, c.stage1)
            )
            q = c.heat_flow(sH.temperature, sC.temperature)
            print(
                f"  {getattr(c,'name',c.material):20s} {sH.name}→{sC.name}: {q:.4f} W"
            )
        print("Radiation:")
        for r in self.radiators:
            if r.stage2 is None:
                continue
            sH, sC = (
                (r.stage1, r.stage2)
                if r.stage1.temperature > r.stage2.temperature
                else (r.stage2, r.stage1)
            )
            q = r.heat_flow(sH.temperature, sC.temperature)
            print(f"  {getattr(r,'name','Rad'):20s} {sH.name}→{sC.name}: {q:.4f} W")
        print("\nStage netQ (should be ~0 for free stages):")
        for s in self.stages:
            print(f"  {s.name:15s}: netQ = {s.net_heat_flow:+.6f} W")
