# utils.py
from __future__ import annotations

import math

INCH = 0.0254  # m

# Keys we treat as linear dimensions (converted with `units`)
_LINEAR_KEYS = {
    "width",
    "thickness",
    "height",
    "length",
    "diameter",
    "outer_dia",
    "inner_dia",
    "dia",
}


def _to_m(val: float | int | None, units: str = "m") -> float | None:
    if val is None:
        return None
    u = units.lower()
    if u in ("m", "meter", "meters", "si"):
        return float(val)
    if u in ("in", "inch", "inches", '"'):
        return float(val) * INCH
    raise ValueError(f"Unknown units '{units}' (use 'm' or 'in').")


def normalize_dims(geom: dict, *, units: str = "m") -> dict:
    """
    Return a copy where:
      • any key ending with '_in' is converted to meters and stripped to base name
      • any key in _LINEAR_KEYS is converted based on `units`
    If both 'width' and 'width_in' are given, '_in' wins (explicit trumps implicit).
    """
    out: dict = {}
    # 1) explicit inch-suffixed keys
    for k, v in list(geom.items()):
        if k.endswith("_in"):
            base = k[:-3]
            out[base] = float(v) * INCH
    # 2) unit-flagged linear keys (only if not already set by *_in)
    for k, v in geom.items():
        if k.endswith("_in"):
            continue
        if k in _LINEAR_KEYS:
            if k not in out:  # don't overwrite explicit *_in
                out[k] = _to_m(v, units)
        else:
            out[k] = v
    return out


# -------------------- geometry helpers --------------------


def cs_area(shape: str, *, units: str = "m", **geom) -> float:
    """
    Cross-sectional area for conduction straps (m^2).
      rect:      width, thickness
      cylinder:  diameter  (or outer_dia)
      tube:      outer_dia, inner_dia
    Accepts *_in keys or `units="in"`.
    """
    g = normalize_dims(geom, units=units)
    shape = shape.lower()
    if shape in ("rect", "rectangle", "strip", "bar"):
        w = g["width"]
        t = g["thickness"]
        return float(w * t)
    if shape in ("cylinder", "wire", "rod"):
        d = g.get("diameter", g.get("outer_dia", g.get("dia")))
        if d is None:
            raise ValueError("cylinder requires diameter/outer_dia/dia")
        return float(math.pi * (d**2) / 4.0)
    if shape in ("tube", "pipe"):
        do = g["outer_dia"]
        di = g["inner_dia"]
        if di >= do:
            raise ValueError("tube inner_dia must be < outer_dia")
        return float(math.pi * (do**2 - di**2) / 4.0)
    raise ValueError(f"Unknown shape '{shape}'")


def surface_area(shape: str, *, units: str = "m", **geom) -> float:
    """
    Exterior radiating area (m^2) for simple solids:
      plate:     width,length  or diameter (disc)
      box:       width,length,height   (all outer faces, closed box)
      cylinder:  diameter,height  (+ 2 end caps)
    Accepts *_in keys or `units="in"`.
    """
    g = normalize_dims(geom, units=units)
    shape = shape.lower()
    if shape in ("plate", "disc", "disk"):
        if "diameter" in g:
            r = g["diameter"] * 0.5
            return float(math.pi * r * r)
        return float(g["width"] * g["length"])
    if shape in ("box", "rect_prism", "brick"):
        w, l, h = g["width"], g["length"], g["height"]
        return float(2.0 * (w * l + w * h + l * h))
    if shape in ("cylinder",):
        d, h = g["diameter"], g["height"]
        r = 0.5 * d
        side = float(math.pi * d * h)
        ends = float(2.0 * math.pi * r * r)
        return side + ends
    raise ValueError(f"Unknown shape '{shape}'")
