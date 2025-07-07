"""
material_db.py
A minimal, dependency-light loader for thermal-conductivity data.

Usage
-----
db = MaterialDatabase("src/cryotherm/data")        # directory of JSON files
k_10K = db.get_k("Al6061", 10)
dk   = db.get_integral("Al6061", 4, 20)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


class MaterialDatabase:
    """Loads both *directory* and *single-file* JSON material tables."""

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __init__(self, path: str | pathlib.Path):
        path = pathlib.Path(path)
        self.materials: dict[str, dict] = {}

        if path.is_dir():
            for jf in path.glob("*.json"):
                self._load_json_file(jf)
        else:
            self._load_json_file(path)

        if not self.materials:
            raise RuntimeError(f"No materials found in '{path}'")

    def get_k(self, material: str, T: float) -> float:
        entry = self._mat(material)
        if entry["model"] == "polylog":
            return self._eval_polylog(entry, T)
        elif entry["model"] == "polynomial":
            return self._eval_polynomial(entry, T)
        elif entry["model"] == "table":
            return self._eval_table(entry, T)
        else:
            raise ValueError(f"Unknown model '{entry['model']}'")

    def get_integral(
        self,
        material: str,
        T1: float,
        T2: float,
        *,
        method: str = "quad",
        n: int = 500,  # only used by trapz
    ) -> float:
        """
        ∫[T1→T2] k(T) dT.

        Parameters
        ----------
        material : str
        T1, T2   : float     Integration limits (K).
        method   : {"quad", "legacy", "trapz"}
            "quad"   – adaptive Gauss-Kronrod (default, highest accuracy)
            "legacy" – matches old Excel ConInt algorithm
            "trapz"  – simple linear-T trapezoidal (n slices)
        n        : int       Number of slices if method="trapz".
        """
        if method == "quad":
            return self._integral_quad(material, T1, T2)

        if method == "legacy":
            return self._integral_legacy(material, T1, T2)

        if method == "trapz":
            return self._integral_trapz(material, T1, T2, n=n)

        raise ValueError(
            f"Unknown method '{method}'. " "Choose 'quad', 'legacy', or 'trapz'."
        )

    # ------------------------------------------------------------------
    # integration back-ends
    # ------------------------------------------------------------------
    def _integral_quad(self, material, T1, T2):
        val, _ = quad(lambda T: self.get_k(material, T), T1, T2, limit=200)
        return val

    def _integral_trapz(self, material, T1, T2, *, n: int):
        grid = np.linspace(T1, T2, n + 1)
        kvals = np.array([self.get_k(material, T) for T in grid])
        return float(np.trapz(kvals, x=grid))

    # ----- Excel-style legacy trapezoid in log10(T) --------------------
    # replicates the old Excel/VBA ConInt algorithm developed by Jeffry Klein
    # at the University of Pennsylvania for the BLAST balloon program

    def _integral_legacy(self, material: str, TempHigh: float, TempLow: float) -> float:
        """
        Excel/VBA-compatible conductivity integral.

        Mirrors the original ConInt macro, including:
        • slice-count heuristic based on k(Thigh)/k(Tlow)
        • log10-spaced grid and trapezoidal rule
        • max 1000 slices
        • ± sign to preserve caller’s integration direction
        """

        # -- ensure TempHigh is the larger temp --
        if TempLow > TempHigh:
            TempHigh, TempLow = TempLow, TempHigh

        # ----- slice-count heuristic (exact VBA) --------------------------
        steps = self.get_k(material, TempHigh) / self.get_k(material, TempLow)
        if steps < 1.0:
            steps = 1.0 / steps
        steps = 30.0 * steps
        if steps > 1000.0:
            steps = 1000.0  # VBA hard cap

        firststep = math.log10(TempHigh)
        laststep = math.log10(TempLow)
        steps = (firststep - laststep) / steps  #  Δlog10T  (positive)

        # ----- integration loop ------------------------------------------
        ConInt = 0.0
        Temp = firststep  # loop variable in log10-space

        while Temp >= laststep - 1e-12:
            temp1 = 10**Temp
            temp2 = 10 ** (Temp - steps)

            yint1 = self.get_k(material, temp1)
            yint2 = self.get_k(material, temp2)
            yavg = 0.5 * (yint1 + yint2)
            dT = temp1 - temp2  # always positive

            ConInt += yavg * dT
            Temp -= steps  # move one slice lower

        return ConInt

    def get_materials(self) -> list[str]:
        """Return a list of available material names."""
        return list(self.materials.keys())

    # ------------------------------------------------------------------ #
    # file parsing helpers
    # ------------------------------------------------------------------ #
    def _load_json_file(self, file_path: pathlib.Path) -> None:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # each top-level key is a material name
        for mat_name, blob in data.items():
            tc = blob.get("thermal_conductivity") or blob
            model = tc["model"].lower()

            if model == "polylog":  # dict of labelled coeffs
                coeff_letters = "abcdefghi"
                coeffs = [tc["coeffs"].get(coeff, 0.0) for coeff in coeff_letters]
                T_min, T_max = tc.get("equation_range", [0.0, float("inf")])

                self.materials[mat_name] = {
                    "model": "polylog",
                    "coeffs": coeffs,
                    "T_min": T_min,
                    "T_max": T_max,
                }

            elif model == "table":
                T_arr = np.array(tc["temperatures"], dtype=float)
                k_arr = np.array(tc["k_values"], dtype=float)
                self.materials[mat_name] = {
                    "model": "table",
                    "T_min": float(T_arr[0]),
                    "T_max": float(T_arr[-1]),
                    "interp": interp1d(
                        T_arr,
                        k_arr,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(k_arr[0], k_arr[-1]),
                    ),
                }
            else:
                raise ValueError(f"Unknown model '{model}' in {file_path}")

    # ------------------------------------------------------------------ #
    # model evaluators
    # ------------------------------------------------------------------ #
    @staticmethod
    def _poly_log10(coeff: list[float], logT: float) -> float:
        """Evaluate Σ c_i * (logT)**i."""
        return sum(c * logT**i for i, c in enumerate(coeff))

    def _eval_polylog(self, entry: dict, T: float) -> float:
        self._range_check(entry, T)
        logT = np.log10(T)
        logk = self._poly_log10(entry["coeffs"], logT)
        return 10.0**logk

    def _eval_polynomial(self, entry: dict, T: float) -> float:
        """Supporting the *old* list-coeff polynomial model."""
        self._range_check(entry, T)
        logT = np.log10(T)
        logk = self._poly_log10(entry["coeffs"], logT)
        return 10.0**logk

    def _eval_table(self, entry: dict, T: float) -> float:
        if T < entry["T_min"]:
            T = entry["T_min"]
        elif T > entry["T_max"]:
            T = entry["T_max"]
        return float(entry["interp"](T))

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    def _mat(self, name: str) -> dict:
        if name not in self.materials:
            raise KeyError(f"Material '{name}' not loaded")
        return self.materials[name]

    @staticmethod
    def _range_check(entry: dict, T: float) -> None:
        # allow 1-part-in-10^9 slack on each bound
        tol = 1e-9 * entry["T_max"]
        if not (entry["T_min"] - tol <= T <= entry["T_max"] + tol):
            raise ValueError(
                f"T={T:g} K outside valid range "
                f"[{entry['T_min']}, {entry['T_max']}] for this fit."
            )

    @staticmethod
    def _flatten_legacy(data: dict) -> dict[str, dict]:
        """Handle original single-file format with 'polynomial' model."""
        out: dict[str, dict] = {}
        for name, info in data.items():
            model = info["model"].lower()
            if model == "polynomial":
                out[name] = {
                    "model": "polynomial",
                    "coeffs": info["coeffs"],
                    "T_min": info.get("T_min", 0.0),
                    "T_max": info.get("T_max", float("inf")),
                }
            elif model == "table":
                T_arr = np.array(info["temperatures"], dtype=float)
                k_arr = np.array(info["k_values"], dtype=float)
                out[name] = {
                    "model": "table",
                    "T_min": float(T_arr[0]),
                    "T_max": float(T_arr[-1]),
                    "interp": interp1d(
                        T_arr,
                        k_arr,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(k_arr[0], k_arr[-1]),
                    ),
                }
        return out
