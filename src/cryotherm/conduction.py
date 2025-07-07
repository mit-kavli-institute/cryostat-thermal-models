# src/cryotherm/conduction.py
from __future__ import annotations

import math
from typing import Any, Literal

from cryotherm.material_db import MaterialDatabase
from cryotherm.utils import cs_area


class Conduction:
    """
    Conductive strap/link between two Stage objects:

        Q_cond =  (A / L) · ∫_{T2→T1} k(T) dT

    Parameters
    ----------
    stage1, stage2 : Stage
        The two stages connected by the strap (heat flows from high-T to low-T).
    length : float
        Strap length in metres.
    area : float
        Cross-sectional area in m².
    number : int
        Number of straps (default 1).
        If > 1, the heat flow is multiplied by this number.
        This is useful for multiple parallel straps.
    material : str
        Material key used by MaterialDatabase.
    mat_db : MaterialDatabase
        Shared instance; lookups are fast so call per slice OK.
    method : {"quad", "legacy", "trapz"}
        Which integral method to use (default "quad").
    """

    def __init__(
        self,
        stage1,
        stage2,
        *,
        length: float,
        material: str,
        mat_db: MaterialDatabase,
        number: int = 1,
        area: float | None = None,
        type: Literal["rect", "cylinder", "tube"] | None = None,
        method: Literal["quad", "legacy", "trapz"] = "quad",
        **geom: Any,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.length = float(length)
        self.material = material
        self.db = mat_db
        self.method = method
        self.number = int(number)

        if area is not None:
            self.area = float(area)
        elif type is not None:
            self.area = cs_area(type, **geom)
        else:
            raise ValueError("Specify `area=` or `type=` + geometry keywords")

    # -----------------------------------------------------------------
    def heat_flow(self, T1: float, T2: float) -> float:
        if math.isclose(T1, T2, rel_tol=1e-14):
            return 0.0

        try:
            dk = self.db.get_integral(self.material, T2, T1, method=self.method)
        except ValueError:  # out of range
            # --- graceful degradation ---------------------------------
            dk = self._safe_integral(T1, T2)
        return (self.area / self.length) * dk * self.number

    # ---------------------------------------------------------------
    def _safe_integral(self, Tlow: float, Thigh: float) -> float:
        """
        Clamp both limits and do a trapezoid (fast, robust).
        Only called when the strict evaluator raised.
        """
        T_min = self.db.materials[self.material]["T_min"]
        T_max = self.db.materials[self.material]["T_max"]
        Tlow_cl = min(max(Tlow, T_min), T_max)
        Thigh_cl = min(max(Thigh, T_min), T_max)
        k1 = self.db.safe_get_k(self.material, Tlow_cl)
        k2 = self.db.safe_get_k(self.material, Thigh_cl)
        return 0.5 * (k1 + k2) * (Thigh_cl - Tlow_cl)
