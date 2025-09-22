from __future__ import annotations

from typing import Any, Literal, Tuple

import numpy as np

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


class WindowRadiation:
    """
    Radiation heat load through an optical window with bandpass characteristics.

    Models selective transmission/reflection based on wavelength-dependent properties.
    Accounts for both transmitted external radiation and re-radiation from the window itself.

    Parameters
    ----------
    stage : object
        The cold stage receiving radiation
    window_temp : float
        Temperature of the window (K)
    passband : tuple[float, float] | None
        Wavelength range (min, max) in meters where window transmits.
        If None, assumes broadband transmission.
    transmission : float
        Transmission coefficient in the passband (0-1)
    reflection : float
        Reflection coefficient outside passband (0-1)
    absorption : float
        Absorption coefficient of window material (0-1)
    diameter : float | None
        Window diameter (m or inches depending on units)
    area : float | None
        Explicit window area (m²) - use this OR diameter
    env_temp : float
        External environment temperature (K), default 300
    view_factor : float
        View factor from window to stage, default 1.0
    units : {"m", "in"}
        Units for diameter dimension
    name : str | None
        Optional name for this radiation element
    """

    _SIGMA = 5.670374419e-8  # W·m⁻²·K⁻⁴
    _h = 6.62607015e-34  # Planck constant (J·s)
    _c = 299792458  # Speed of light (m/s)
    _k = 1.380649e-23  # Boltzmann constant (J/K)

    def __init__(
        self,
        stage,
        *,
        window_temp: float,
        passband: Tuple[float, float] | None = None,
        transmission: float = 0.9,
        reflection: float = 0.9,
        absorption: float = 0.1,
        diameter: float | None = None,
        area: float | None = None,
        env_temp: float = 300.0,
        view_factor: float = 1.0,
        units: Literal["m", "in"] = "m",
        name: str | None = None,
    ):
        self.stage = stage
        self.window_temp = float(window_temp)
        self.env_temp = float(env_temp)
        self.name = name or "WindowRadiation"
        self.F = float(view_factor)

        # Optical properties
        self.passband = passband  # (lambda_min, lambda_max) in meters
        self.tau = float(transmission)  # transmission in passband
        self.rho = float(reflection)  # reflection outside passband
        self.alpha = float(absorption)  # absorption coefficient

        # Validate optical properties
        if not 0 <= self.tau <= 1:
            raise ValueError("transmission must be between 0 and 1")
        if not 0 <= self.rho <= 1:
            raise ValueError("reflection must be between 0 and 1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("absorption must be between 0 and 1")

        # Compute window area
        if area is not None:
            self.area = float(area)
        elif diameter is not None:
            # Use surface_area for consistency with Radiation class
            geom = {"diameter": diameter}
            geom_m = normalize_dims(geom, units=units)
            self.area = surface_area("plate", **geom_m)
        else:
            raise ValueError("Specify either `area=` or `diameter=`")

    def planck_radiance(self, wavelength: float, T: float) -> float:
        """
        Planck's law for spectral radiance (W·m⁻²·sr⁻¹·m⁻¹).

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        T : float
            Temperature in Kelvin
        """
        if T <= 0:
            return 0.0

        # Avoid numerical issues
        x = (self._h * self._c) / (wavelength * self._k * T)
        if x > 700:  # exp(700) would overflow
            return 0.0

        num = 2 * self._h * self._c**2
        denom = wavelength**5 * (np.exp(x) - 1)
        return num / denom

    def integrated_power(
        self, T: float, lambda_min: float = None, lambda_max: float = None
    ) -> float:
        """
        Integrate Planck's law over wavelength range to get total power.

        Uses Stefan-Boltzmann for full spectrum, or numerical integration for bands.

        Parameters
        ----------
        T : float
            Temperature (K)
        lambda_min, lambda_max : float | None
            Wavelength range (m). If both None, returns full blackbody power.
        """
        if lambda_min is None and lambda_max is None:
            # Full spectrum: use Stefan-Boltzmann
            return self._SIGMA * T**4

        # For bandpass: use simplified rectangular approximation
        # (For production code, use scipy.integrate.quad for accuracy)
        if lambda_min is None:
            lambda_min = 1e-9  # 1 nm
        if lambda_max is None:
            lambda_max = 1e-3  # 1 mm

        # Peak wavelength from Wien's law
        lambda_peak = 2.898e-3 / T  # meters

        # Simple approximation: evaluate at peak if in range, else at midpoint
        if lambda_min <= lambda_peak <= lambda_max:
            lambda_eval = lambda_peak
        else:
            lambda_eval = (lambda_min + lambda_max) / 2

        # Approximate as rectangular spectrum
        bandwidth = lambda_max - lambda_min
        spectral_radiance = self.planck_radiance(lambda_eval, T)

        # Convert to power (integrate over solid angle: multiply by pi for Lambertian)
        return np.pi * spectral_radiance * bandwidth

    def heat_flow(self, T_stage: float) -> float:
        """
        Calculate net heat flow to the cold stage through the window.

        Components:
        1. Transmitted external radiation (in passband)
        2. Blocked/reflected external radiation (outside passband)
        3. Thermal emission from window itself

        Parameters
        ----------
        T_stage : float
            Temperature of the cold stage (K)

        Returns
        -------
        float
            Net heat flow to stage (W), positive = heating
        """
        # External radiation incident on window
        P_ext_total = self._SIGMA * self.env_temp**4 * self.area * self.F

        if self.passband is not None:
            # Calculate fraction of blackbody power in passband
            lambda_min, lambda_max = self.passband
            P_ext_band = (
                self.integrated_power(self.env_temp, lambda_min, lambda_max)
                * self.area
                * self.F
            )
            P_ext_out = P_ext_total - P_ext_band

            # Transmitted power in passband
            P_transmitted = self.tau * P_ext_band

            # Absorbed power (heats window)
            P_absorbed_band = (1 - self.tau - self.rho) * P_ext_band
            P_absorbed_out = (1 - self.rho) * P_ext_out
            P_absorbed_total = P_absorbed_band + P_absorbed_out
        else:
            # Broadband window
            P_transmitted = self.tau * P_ext_total
            P_absorbed_total = self.alpha * P_ext_total

        # Window emission (both sides, but only one side faces stage)
        P_window_emission = (
            self.alpha * self._SIGMA * self.window_temp**4 * self.area * self.F
        )

        # Stage back-radiation (assuming stage is grey body with emissivity ~1)
        P_stage_emission = self._SIGMA * T_stage**4 * self.area * self.F

        # Net heat flow to stage
        Q_net = P_transmitted + P_window_emission - P_stage_emission

        return Q_net

    def heat_flow_simple(self, T_stage: float) -> float:
        """
        Simplified heat flow calculation assuming window is at steady state.

        Good approximation when window equilibrates quickly.
        """
        if self.passband is not None:
            # Effective emissivity based on passband
            # This is a simplified model
            lambda_min, lambda_max = self.passband

            # Estimate fraction of 300K blackbody in passband
            # (crude approximation - improve for production)
            lambda_peak_300K = 2.898e-3 / 300  # ~10 microns

            if lambda_min <= lambda_peak_300K <= lambda_max:
                # Peak is in passband
                f_band = 0.8  # Most power is near peak
            else:
                # Peak is outside passband
                f_band = 0.1  # Small fraction in band

            # Effective transmittance
            tau_eff = self.tau * f_band + (1 - self.rho) * (1 - f_band)
        else:
            tau_eff = self.tau

        # Net radiation using effective parameters
        Q_net = (
            tau_eff * self._SIGMA * self.area * self.F * (self.env_temp**4 - T_stage**4)
        )

        return Q_net


# Example usage functions
def create_mlr_window(stage, diameter_in=2.0, window_temp=77):
    """
    Create a multi-layer insulation window for IR blocking.
    Typical MLI window: reflects IR, transmits visible/NIR.
    """
    return WindowRadiation(
        stage,
        diameter=diameter_in,
        units="in",
        window_temp=window_temp,
        passband=(400e-9, 2e-6),  # 400nm to 2μm
        transmission=0.85,
        reflection=0.95,  # High IR reflectivity
        absorption=0.05,
        name="MLI_Window",
    )


def create_polyethylene_window(stage, diameter_in=3.0, window_temp=50):
    """
    Create a polyethylene window for FIR/mm-wave transmission.
    """
    return WindowRadiation(
        stage,
        diameter=diameter_in,
        units="in",
        window_temp=window_temp,
        passband=(10e-6, 3e-3),  # 10μm to 3mm
        transmission=0.90,
        reflection=0.08,
        absorption=0.02,
        name="Polyethylene_Window",
    )


def create_sapphire_window(stage, diameter_in=1.0, window_temp=77):
    """
    Create a sapphire window for UV-NIR transmission.
    """
    return WindowRadiation(
        stage,
        diameter=diameter_in,
        units="in",
        window_temp=window_temp,
        passband=(150e-9, 5e-6),  # 150nm to 5μm
        transmission=0.86,
        reflection=0.12,
        absorption=0.02,
        name="Sapphire_Window",
    )
