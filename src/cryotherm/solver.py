# src/cryotherm/solver.py
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import least_squares, root


class ThermalModel:
    """
    Assemble stages + conductive & radiative links
    and solve for steady-state (`netQ = 0`) temperatures.

    Stages flagged `fixed=True` are held at their initial temperature.
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

        self.free = [s for s in stages if not s.fixed]

    # -----------------------------------------------------------------
    # residuals
    # -----------------------------------------------------------------
    def _balance_equations(self, temps: np.ndarray) -> np.ndarray:
        """
        Residual vector:
            • fridge stage i :  T_i − T_target(load_i)  = 0
            • plain  stage i :  netQ_i                  = 0
        """

        # 0) assign trial temperatures to free stages
        for s, T in zip(self.free, temps, strict=True):
            s.temperature = T

        # 1) zero tallies
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

        # 4) build residuals
        residuals = []
        for s in self.free:
            if s.has_fridge:
                T_target = s.target_temp_for_load(s._q_in)
                residuals.append(s.temperature - T_target)
            else:
                residuals.append(s._q_in)

            # diagnostics
            """s.net_heat_flow = s._q_in - (
                s.cooling_power(s.temperature) if s.has_fridge else 0.0
            )"""
            s.net_heat_flow = s._q_in

        return np.array(residuals, dtype=float)

    # -----------------------------------------------------------------
    # public solver
    # -----------------------------------------------------------------

    def solve(self, tol: float = 1e-6):
        # ---- initial guess ------------------------------------------------
        T0 = []
        lo = []
        hi = []
        for s in self.free:
            Tmin, Tmax = s.fridge_bounds
            Tguess = np.clip(s.temperature, Tmin, Tmax)
            if s.has_fridge and not (Tmin <= s.temperature <= Tmax):
                # midpoint of the fridge curve is a robust start
                Tguess = 0.5 * (Tmin + Tmax)
            T0.append(Tguess)
            lo.append(Tmin)
            hi.append(Tmax)
        T0 = np.array(T0)
        bounds = (np.array(lo), np.array(hi))

        # ---- bounded least-squares (robust for monotone curves) -----------
        sol = least_squares(
            self._balance_equations,
            T0,
            bounds=bounds,
            xtol=tol,
            ftol=tol,
            gtol=tol,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        for s, T in zip(self.free, sol.x, strict=True):
            s.temperature = T
        return sol
