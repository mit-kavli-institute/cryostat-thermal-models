from __future__ import annotations

import math
from typing import Any, Literal

from cryotherm.utils import normalize_dims, surface_area


class Radiation:
    """
    Grey-body radiation between surfaces.

    New: geometry in inches via `units="in"` or *_in keywords.
    If you pass a shape, area is computed; else use explicit `area`.
    """

    _SIGMA = 5.670374419e-8  # W·m⁻²·K⁻⁴

    def __init__(
        self,
        stage1,
        stage2=None,
        *,
        emissivity1: float,
        area: float | None = None,
        view_factor: float = 1.0,
        env_temp: float = 300.0,
        emissivity2: float | None = None,
        type: Literal["cylinder", "plate", "box"] | None = None,
        units: Literal["m", "in"] = "m",
        name: str | None = None,
        **geom: Any,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.name = name or "Radiation"
        self.F = float(view_factor)
        self.env_temp = float(env_temp)

        eps1 = float(emissivity1)
        eps2 = float(emissivity2) if emissivity2 is not None else None
        self.eps = self.effective_emissivity(eps1, eps2)

        if area is not None:
            self.area = float(area)  # assume m^2 if explicit
        elif type is not None:
            geom_m = normalize_dims(geom, units=units)
            self.area = surface_area(type, **geom_m)
        else:
            raise ValueError("Specify `area=` or `type=` + geometry keywords")

    def heat_flow(self, T1: float, T2: float | None = None) -> float:
        if T2 is None:
            T2 = self.env_temp
        return self.eps * self._SIGMA * self.area * self.F * (T1**4 - T2**4)

    @staticmethod
    def effective_emissivity(eps1: float, eps2: float | None = None) -> float:
        if eps2 is None:
            return eps1
        return eps1 * eps2 / (eps1 + eps2 - eps1 * eps2)
