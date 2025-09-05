# src/cryotherm/sizing.py
from __future__ import annotations

import numpy as np

from cryotherm.material_db import MaterialDatabase


def size_strap_conductance(
    hot_stage,
    *,
    cold_stage_target: float,
    load_W: float,
    material: str,
    length: float,
    mat_db: MaterialDatabase,
    method: str = "quad",  # same options as get_integral
) -> float:
    """
    Return the conductance G (W K⁻¹) required between *hot_stage*
    and a *cold_stage* you want to clamp at `cold_stage_target`.

    Assumptions
    -----------
    • Only *one* strap between the two stages.
    • All other loads on the cold stage are included in `load_W`
      (positive = heat that must be removed).
    • Hot-stage temperature is *known* (fixed or already solved).

    The strap can later be realised as either:
        - fixed conductance  (conductance=G_req), or
        - geometry+material  with area = G_req·L / k_avg.
    """
    T_hot = hot_stage.temperature
    T_cold = float(cold_stage_target)

    if T_hot <= T_cold + 1e-6:
        raise ValueError("Hot-stage temperature must exceed cold-stage target.")

    # average k over the anticipated range (more stable than k at midpoint)
    dk = mat_db.get_integral(material, T_cold, T_hot, method=method)
    k_avg = dk / (T_hot - T_cold)

    # conductance needed to carry the load
    G_req = load_W / (T_hot - T_cold)

    # some users like to see equivalent area
    size_strap_conductance.k_avg = k_avg
    return G_req
