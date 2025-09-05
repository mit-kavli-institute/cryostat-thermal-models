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
    def __init__(
        self, path: str | pathlib.Path, *, recursive: bool = False, debug: bool = False
    ):
        path = pathlib.Path(path)
        self.materials: dict[str, dict] = {}
        self._sources: dict[str, pathlib.Path] = {}  # material -> json file
        self._load_errors: list[tuple[pathlib.Path, str, str]] = (
            []
        )  # (file, material, message)
        self._duplicates: list[tuple[str, pathlib.Path, pathlib.Path]] = (
            []
        )  # (material, old_file, new_file)
        self._debug = bool(debug)

        files: list[pathlib.Path]
        if path.is_dir():
            files = (
                list(path.rglob("*.json")) if recursive else list(path.glob("*.json"))
            )
        else:
            files = [path]

        for jf in files:
            self._load_json_file(jf)

        if not self.materials:
            raise RuntimeError(f"No materials found in '{path}'")

    def get_k(self, material: str, T: float) -> float:
        entry = self._mat(material)
        if entry["model"] == "polylog":
            return self._eval_polylog(entry, T)
        elif entry["model"] == "nist-copperfit":
            return self._eval_nist_copperfit(entry, T)
        elif entry["model"] == "table":
            return self._eval_table(entry, T)
        else:
            raise ValueError(f"Unknown model '{entry['model']}'")

    def safe_get_k(self, material: str, T: float) -> float:
        """
        Like get_k() but CLAMPS temperature to the material’s valid range
        instead of raising.  Handy for numerics when the solver wanders
        outside  the table / fit domain.
        """
        entry = self._mat(material)
        T_clamped = min(max(T, entry["T_min"]), entry["T_max"])
        return self.get_k(material, T_clamped)  # now in‐range

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

    def report(self) -> None:
        print(f"Loaded materials: {self.get_materials()}")
        if self._duplicates:
            print("\nDuplicates (new overwrote old):")
            for name, oldf, newf in self._duplicates:
                print(f" - {name}: {oldf.name} -> {newf.name}")
        if self._load_errors:
            print("\nErrors:")
            for f, name, msg in self._load_errors:
                print(f" - {f.name} :: {name}: {msg}")

    # ------------------------------------------------------------------ #
    # file parsing helpers
    # ------------------------------------------------------------------ #
    def _load_json_file(self, file_path: pathlib.Path) -> None:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._load_errors.append((file_path, "<file>", f"JSON load failed: {e!r}"))
            if self._debug:
                print(f"[ERROR] {file_path}: {e!r}")
            return

        for mat_name, blob in data.items():
            try:
                tc = blob.get("thermal_conductivity") or blob
                model_raw = tc["model"]
            except Exception as e:
                self._load_errors.append(
                    (file_path, mat_name, f"Missing/invalid keys: {e!r}")
                )
                if self._debug:
                    print(f"[ERROR] {file_path}:{mat_name} -> {e!r}")
                continue

            # Accept a few aliases
            model = str(model_raw).lower().strip()
            aliases = {"polynomial": "polylog", "csv": "table"}
            model = aliases.get(model, model)

            try:
                if model == "polylog":
                    coeff_letters = "abcdefghi"
                    coeffs = [tc["coeffs"].get(letter, 0.0) for letter in coeff_letters]
                    T_min, T_max = tc.get("equation_range", [0.0, float("inf")])
                    entry = {
                        "model": "polylog",
                        "coeffs": coeffs,
                        "T_min": float(T_min),
                        "T_max": float(T_max),
                    }

                elif model == "nist-copperfit":
                    coeff_letters = "abcdefghi"
                    coeffs = [tc["coeffs"].get(letter, 0.0) for letter in coeff_letters]
                    T_min, T_max = tc.get("equation_range", [0.0, float("inf")])
                    entry = {
                        "model": "nist-copperfit",
                        "coeffs": coeffs,
                        "T_min": float(T_min),
                        "T_max": float(T_max),
                    }

                elif model == "table":
                    # Option A: external CSV
                    if "file" in tc:
                        csv_path = pathlib.Path(tc["file"])
                        if not csv_path.is_absolute():
                            csv_path = file_path.parent / csv_path
                        delimiter = tc.get("delimiter", ",")
                        skip_rows = int(tc.get("skip_rows", 0))
                        col_T = int(tc.get("col_T", 0))
                        col_k = int(tc.get("col_k", 1))
                        if not csv_path.exists():
                            raise FileNotFoundError(f"CSV not found: {csv_path}")
                        data_arr = np.loadtxt(
                            csv_path,
                            delimiter=delimiter,
                            skiprows=skip_rows,
                            usecols=(col_T, col_k),
                            ndmin=2,
                            comments="#",
                        )
                        T_arr = np.asarray(data_arr[:, 0], dtype=float)
                        k_arr = np.asarray(data_arr[:, 1], dtype=float)
                    else:
                        # Option B: inline arrays
                        T_arr = np.array(tc["temperatures"], dtype=float)
                        k_arr = np.array(tc["k_values"], dtype=float)

                    if T_arr.size != k_arr.size or T_arr.size < 2:
                        raise ValueError("need ≥2 rows and equal lengths for T and k")

                    # sort + dedupe T
                    order = np.argsort(T_arr)
                    T_arr, k_arr = T_arr[order], k_arr[order]
                    T_arr, idx = np.unique(T_arr, return_index=True)
                    k_arr = k_arr[idx]

                    entry = {
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
                    raise ValueError(f"Unknown model '{model}'")

                # store, noting duplicates
                if mat_name in self.materials:
                    self._duplicates.append(
                        (mat_name, self._sources[mat_name], file_path)
                    )
                    if self._debug:
                        print(
                            f"[WARN] duplicate '{mat_name}' -> {file_path.name} overrides {self._sources[mat_name].name}"
                        )
                self.materials[mat_name] = entry
                self._sources[mat_name] = file_path

            except Exception as e:
                self._load_errors.append(
                    (file_path, mat_name, f"{e.__class__.__name__}: {e}")
                )
                if self._debug:
                    print(
                        f"[ERROR] {file_path}:{mat_name} -> {e.__class__.__name__}: {e}"
                    )

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
        return float(10.0**logk)

    def _eval_nist_copperfit(self, entry: dict, T: float) -> float:
        """
        NIST OFHC copper fit (W/m·K):
        log10(k) = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2) /
                    (1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        => k = 10 ** (numerator / denominator)
        Coefficients are in the order a, b, c, d, e, f, g, h, i.
        ref: https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm
        """
        self._range_check(entry, T)
        a, b, c, d, e, f, g, h, i = entry["coeffs"]

        rt = np.sqrt(T)
        t32 = T * rt
        t2 = T * T

        num = a + c * rt + e * T + g * t32 + i * t2
        den = 1.0 + b * rt + d * T + f * t32 + h * t2

        log10k = num / den
        # Optional safety clamp if you want to guard against bad inputs:
        # log10k = float(np.clip(log10k, -12, 12))  # k in ~[1e-12, 1e12] W/mK

        return float(10.0**log10k)

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
