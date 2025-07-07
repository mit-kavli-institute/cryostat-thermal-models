# src/cryotherm/solver.py
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import root


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
        # update free-stage temps
        for s, T in zip(self.free, temps, strict=True):
            s.temperature = T

        # zero all netQ
        for s in self.stages:
            s.net_heat_flow = 0.0

        # conduction
        for link in self.conductors:
            T1 = link.stage1.temperature
            T2 = link.stage2.temperature
            q = link.heat_flow(T1, T2)
            link.stage1.net_heat_flow -= q
            link.stage2.net_heat_flow += q

        # radiation
        for rad in self.radiators:
            T1 = rad.stage1.temperature
            T2 = rad.stage2.temperature if rad.stage2 else None
            q = rad.heat_flow(T1, T2)
            rad.stage1.net_heat_flow -= q
            if rad.stage2:
                rad.stage2.net_heat_flow += q

        # fridge & external loads
        for s in self.stages:
            s.net_heat_flow -= s.cooling_power(s.temperature)
            s.net_heat_flow += s.external_load(s.temperature)

        return np.array([s.net_heat_flow for s in self.free])

    # -----------------------------------------------------------------
    # public solver
    # -----------------------------------------------------------------
    def solve(self, tol: float = 1e-6, method: str = "hybr"):
        T0 = np.array([s.temperature for s in self.free])
        sol = root(self._balance_equations, T0, method=method, tol=tol)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # propagate converged temps
        for s, T in zip(self.free, sol.x, strict=True):
            s.temperature = T
        return sol
