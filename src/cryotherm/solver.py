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

        # free temps: not fixed and not heater-sized
        self.temp_vars = [
            s for s in stages if (not s.fixed) and s.target_temperature is None
        ]
        # heater-sized: target_temperature is specified
        self.heat_vars = [s for s in stages if s.target_temperature is not None]

    # -----------------------------------------------------------------
    def _residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Unknown vector x = [free_temperatures..., heater_powers...]
        """
        N_t = len(self.temp_vars)

        # 0) assign trial values
        for i, s in enumerate(self.temp_vars):
            s.temperature = x[i]

        for j, s in enumerate(self.heat_vars, start=N_t):
            s._heater_W = float(x[j])  # <— only the heater is unknown
            s.temperature = s.target_temperature

        # 1) zero inbound tallies with externals (baseline + heater)
        for s in self.stages:
            s._q_in = s.external_load(s.temperature)

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

        # 4) residuals
        res = []

        # a) temperature unknowns
        for s in self.temp_vars:
            if s.has_fridge:
                T_target = s.target_temp_for_load(s._q_in)
                res.append(s.temperature - T_target)
            else:
                res.append(s._q_in)  # want 0 W net

        # b) heater-power unknowns (temperature is clamped)
        for s in self.heat_vars:
            res.append(s._q_in)  # want 0 W net at target T

        # diagnostics
        for s in self.stages:
            s.net_heat_flow = s._q_in

        return np.array(res, dtype=float)

    @staticmethod
    def _initial_heater_guess(stage) -> float:
        # start from whatever the user last set (default 0)
        return float(getattr(stage, "_heater_W", 0.0))

    # -----------------------------------------------------------------
    def solve(self, tol: float = 1e-6):
        # initial guess and bounds
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
            P0 = self._initial_heater_guess(s)
            x0.append(P0)
            lo.append(0.0)
            hi.append(1e6)  # 0…1 MW

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

        # Write back final heater powers (temps are already set in residuals)
        N_t = len(self.temp_vars)
        for j, s in enumerate(self.heat_vars, start=N_t):
            s._heater_W = float(sol.x[j])

        return sol

    def report(self):
        print("\n=== Balance at current temperatures ===")
        for s in self.stages:
            extra = ""
            if s in self.heat_vars:
                extra = f"  (heater={s.heater_power:.4f} W, baseline={s._baseline_load(s.temperature):.4f} W)"
            print(f"{s.name:15s}: T={s.temperature:7.2f} K{extra}")
        print("\nConduction:")
        for c in self.conductors:
            sH, sC = (
                (c.stage1, c.stage2)
                if c.stage1.temperature > c.stage2.temperature
                else (c.stage2, c.stage1)
            )
            q = c.heat_flow(sH.temperature, sC.temperature)
            print(
                f"  {getattr(c, 'name', c.material):20s} {sH.name}→{sC.name}: {q:.4f} W"
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
            print(
                f"  {getattr(r, 'name', 'Radiation'):20s} {sH.name}→{sC.name}: {q:.4f} W"
            )
        print("\nStage netQ (should be ~0 for free stages):")
        for s in self.stages:
            print(f"  {s.name:15s}: netQ = {s.net_heat_flow:+.6f} W")
