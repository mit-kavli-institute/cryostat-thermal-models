# src/cryotherm/conduction.py
from __future__ import annotations

import math
from typing import Any, Literal

from cryotherm.material_db import MaterialDatabase
from cryotherm.utils import _to_m, cs_area, normalize_dims


class Conduction:
    """
    Conductive link between two stages.

        Q = (A/L) * ∫_{Tc}^{Th} k(T) dT   (strap path, W)

    Optional extras:
      • conductance (W/K): parallel path added to the strap
      • contact_conductance (W/K): series contact resistance with the strap

    Geometry/units niceties:
      • `units="in"` converts all linear dims (length + geometry) from inches
      • or pass explicit *_in keywords (e.g., width_in=0.5)
      • if `area=` is supplied, you may set `area_units="in2"`
    """

    def __init__(
        self,
        stage1,
        stage2,
        *,
        length: float | None = None,
        material: str | None = None,
        mat_db: MaterialDatabase | None = None,
        number: int = 1,
        area: float | None = None,
        area_units: Literal["m2", "in2"] = "m2",
        type: Literal["rect", "cylinder", "tube"] | None = None,
        method: Literal["quad", "legacy", "trapz"] = "quad",
        name: str | None = None,
        units: Literal["m", "in"] = "m",
        conductance: float | None = None,  # parallel (W/K)
        contact_conductance: float | None = None,  # series with strap (W/K)
        **geom: Any,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.material = material
        self.db = mat_db
        self.method = method
        self.number = int(number)
        self.name = name or (f"{material} strap" if material else "strap")

        # optional extra paths
        self.G_parallel = float(conductance) if conductance is not None else None
        self.G_contact = (
            float(contact_conductance) if contact_conductance is not None else None
        )

        # --- resolve length ---
        if length is None and "length_in" in geom:
            length = geom.pop("length_in")
            units = "in"
        if length is None:
            raise ValueError("Conduction needs `length` (or `length_in`).")
        self.length = float(_to_m(length, units))

        # --- resolve area ---
        if area is not None:
            self.area = (
                float(area) * (0.0254**2) if area_units == "in2" else float(area)
            )
        else:
            if type is None:
                raise ValueError("Specify `area=` or `type=` + geometry keywords.")
            geom_m = normalize_dims(geom, units=units)
            self.area = cs_area(type, **geom_m)

    # ------------------------------------------------------------------
    def _strap_integral(self, T_hot: float, T_cold: float) -> float:
        """
        Return ∫_{Tc}^{Th} k(T) dT (units: W/m).
        Falls back to a clamped trapezoid if the database range is exceeded.
        """
        try:
            return self.db.get_integral(
                self.material, T_cold, T_hot, method=self.method
            )
        except Exception:
            # graceful fallback (clamped trapezoid)
            T_min = self.db.materials[self.material]["T_min"]
            T_max = self.db.materials[self.material]["T_max"]
            Tc = min(max(T_cold, T_min), T_max)
            Th = min(max(T_hot, T_min), T_max)
            k1 = self.db.safe_get_k(self.material, Tc)
            k2 = self.db.safe_get_k(self.material, Th)
            return 0.5 * (k1 + k2) * (Th - Tc)

    def _strap_heat(self, T_hot: float, T_cold: float) -> float:
        """
        Heat through the strap path alone (W):
            Q_strap = (A/L) * ∫ k(T) dT
        """
        if (self.material is None) or math.isclose(T_hot, T_cold, rel_tol=1e-14):
            return 0.0
        dk = self._strap_integral(T_hot, T_cold)  # W/m
        return (self.area / self.length) * dk  # W

    # ------------------------------------------------------------------
    def heat_flow(self, T1: float, T2: float) -> float:
        """
        Positive when heat flows from stage1 → stage2.
        Combines:
          • non-linear strap path (integral k(T))
          • optional series contact (W/K)
          • optional parallel path  (W/K)
        """
        if math.isclose(T1, T2, rel_tol=1e-14):
            return 0.0

        # orient hot/cold and set sign so result is +ve for stage1→stage2
        Th, Tc = (T1, T2) if T1 > T2 else (T2, T1)
        sign = 1.0 if T1 > T2 else -1.0
        dT = Th - Tc

        # --- strap path gives HEAT directly (W), no extra ΔT factor
        Q_strap = self._strap_heat(Th, Tc)

        # Convert that strap path to an *effective conductance* at this (Th,Tc)
        # so we can combine with fixed G in series/parallel.
        G_strap_eff = Q_strap / dT if dT > 0 else 0.0  # W/K

        # --- series contact with strap only
        if self.G_contact is not None and G_strap_eff > 0.0:
            G_series = 1.0 / (1.0 / G_strap_eff + 1.0 / self.G_contact)
        else:
            G_series = G_strap_eff

        # Heat through the strap+contact branch
        Q_series = G_series * dT

        # --- add optional parallel conductance path (W/K)
        Q_parallel = (self.G_parallel or 0.0) * dT

        Q_total = (Q_series + Q_parallel) * self.number  # W
        return sign * Q_total
