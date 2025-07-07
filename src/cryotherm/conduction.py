# src/cryotherm/conduction.py
from __future__ import annotations

import math
from typing import Any

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
        area: float,
        number: int = 1,
        material: str,
        mat_db: MaterialDatabase,
        method: str = "quad",
        **geom: Any,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.length = float(length)
        self.number = int(number)
        self.material = material
        self.db = mat_db
        self.method = method

        if area is not None:
            self.area = float(area)
        elif type is not None:
            self.area = cs_area(type, **geom)
        else:
            raise ValueError("Specify `area=` or `type=` + geometry keywords")

    # -----------------------------------------------------------------
    def heat_flow(self, T1: float, T2: float) -> float:
        """
        Positive when heat flows from `stage1` → `stage2`.
        """
        if math.isclose(T1, T2, rel_tol=1e-14, abs_tol=0.0):
            return 0.0

        dk = self.db.get_integral(
            self.material, T2, T1, method=self.method
        )  # note order: lower limit, upper limit
        return (self.area / self.length) * dk * self.number
