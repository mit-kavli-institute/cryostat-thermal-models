# src/cryotherm/radiation.py
from __future__ import annotations

import math
from typing import Any

from cryotherm.utils import surf_area


class Radiation:
    """
    Simple grey-body radiative link:

        Q_rad = ε · σ · A · F_12 · (T1⁴ − T2⁴)

    If `stage2` is None, the link points to a fixed-temperature
    environment (`env_temp`).

    Parameters
    ----------
    emissivity : float   (0‥1)
    area       : float   (m²)
    view_factor: float   (dimensionless, 0‥1)
    env_temp   : float   (K)  when stage2 is None
    """

    _SIGMA = 5.670374419e-8  # W · m⁻² · K⁻⁴

    def __init__(
        self,
        stage1,
        stage2=None,
        *,
        emissivity1: float,
        area: float,
        view_factor: float = 1.0,
        env_temp: float = 300.0,
        emissivity2: float | None = None,
        **geom: Any,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.eps1 = float(emissivity1)
        self.eps2 = float(emissivity2) if emissivity2 is not None else self.eps
        self.F = float(view_factor)
        self.env_temp = float(env_temp)
        # calculate the effective emissivity
        self.eps = self.effective_emissivity(self.eps1, self.eps2)

        if area is not None:
            self.area = float(area)
        elif type is not None:
            self.area = surf_area(type, **geom)
        else:
            raise ValueError("Specify `area=` or `type=` + dimensions")

    # -----------------------------------------------------------------
    def heat_flow(self, T1: float, T2: float | None = None) -> float:
        """Positive when heat leaves `stage1`."""
        if T2 is None:
            T2 = self.env_temp
        return self.eps * self._SIGMA * self.area * self.F * (T1**4 - T2**4)

    # -----------------------------------------------------------------
    @staticmethod
    def effective_emissivity(eps1: float, eps2: float | None = None) -> float:
        """
        Calculate the effective emissivity of two surfaces.

        If `eps2` is None, it is assumed that the second surface
        is a perfect absorber (ε = 0).
        """
        if eps2 is None:
            return 1
        return eps1 * eps2 / (eps1 + eps2 - eps1 * eps2)
