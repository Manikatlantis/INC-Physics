"""Numerical estimators used for finite-size critical-temperature analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Chebyshev
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class HeatCapacityCurve:
    temperatures: NDArray[np.float64]
    heat_capacity_per_site: NDArray[np.float64]
    peak_temperature: float
    peak_heat_capacity_per_site: float
    fit_rms_log_z: float
    polynomial_degree: int


def heat_capacity_from_log_z(
    temperatures: ArrayLike,
    log_z_values: ArrayLike,
    n_sites: int,
    polynomial_degree: int = 10,
    dense_points: int = 2_000,
    peak_interval: tuple[float, float] = (1.8, 2.8),
) -> HeatCapacityCurve:
    """Smooth log Z(beta) with a Chebyshev fit and differentiate twice.

    The thermodynamic identity gives total heat capacity
    beta^2 d^2(log Z)/d beta^2.  Division by ``n_sites`` returns the intensive
    convention used elsewhere in this repository; the peak location is
    unaffected by that division.
    """

    temperature_array = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    log_z_array = np.asarray(log_z_values, dtype=np.float64).reshape(-1)
    if temperature_array.shape != log_z_array.shape or temperature_array.size < 4:
        raise ValueError("temperatures and log_z_values must be equally sized with >=4 points")
    if np.any(temperature_array <= 0) or n_sites <= 0:
        raise ValueError("temperatures and n_sites must be positive")
    if not (peak_interval[0] < peak_interval[1]):
        raise ValueError("peak interval must be ordered")

    beta = 1.0 / temperature_array
    order = np.argsort(beta)
    beta = beta[order]
    log_z_array = log_z_array[order]
    degree = min(int(polynomial_degree), beta.size - 1)
    fit = Chebyshev.fit(beta, log_z_array, degree, domain=[beta[0], beta[-1]])
    fitted_at_input = fit(beta)
    fit_rms = float(np.sqrt(np.mean((fitted_at_input - log_z_array) ** 2)))
    second_derivative = fit.deriv(2)

    def heat_capacity_at_temperature(temperature: float) -> float:
        beta_value = 1.0 / temperature
        return float(beta_value**2 * second_derivative(beta_value) / n_sites)

    optimization = minimize_scalar(
        lambda temperature: -heat_capacity_at_temperature(float(temperature)),
        bounds=peak_interval,
        method="bounded",
        options={"xatol": 1e-12},
    )
    dense_temperatures = np.linspace(
        float(np.min(temperature_array)),
        float(np.max(temperature_array)),
        dense_points,
        dtype=np.float64,
    )
    dense_beta = 1.0 / dense_temperatures
    dense_heat_capacity = dense_beta**2 * second_derivative(dense_beta) / n_sites
    return HeatCapacityCurve(
        temperatures=dense_temperatures,
        heat_capacity_per_site=np.asarray(dense_heat_capacity, dtype=np.float64),
        peak_temperature=float(optimization.x),
        peak_heat_capacity_per_site=-float(optimization.fun),
        fit_rms_log_z=fit_rms,
        polynomial_degree=degree,
    )


def local_quadratic_peak(x_values: ArrayLike, y_values: ArrayLike, half_window: int = 2) -> float:
    """Estimate a smooth peak by a local quadratic around the largest sample."""

    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing and match at least three y values")
    maximum_index = int(np.argmax(y))
    start = max(0, maximum_index - half_window)
    stop = min(x.size, maximum_index + half_window + 1)
    if stop - start < 3:
        return float(x[maximum_index])
    coefficients = np.polyfit(x[start:stop], y[start:stop], deg=2)
    quadratic, linear, _ = coefficients
    if quadratic >= 0:
        return float(x[maximum_index])
    vertex = -linear / (2.0 * quadratic)
    if not (x[start] <= vertex <= x[stop - 1]):
        return float(x[maximum_index])
    return float(vertex)


def linear_crossing(
    x_values: ArrayLike,
    first_curve: ArrayLike,
    second_curve: ArrayLike,
    interval: tuple[float, float],
) -> float:
    """Find the first linearly interpolated curve crossing inside an interval."""

    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    first = np.asarray(first_curve, dtype=np.float64).reshape(-1)
    second = np.asarray(second_curve, dtype=np.float64).reshape(-1)
    if x.shape != first.shape or x.shape != second.shape or np.any(np.diff(x) <= 0):
        raise ValueError("curves must match a strictly increasing x grid")
    difference = first - second
    for index in range(x.size - 1):
        if x[index + 1] < interval[0] or x[index] > interval[1]:
            continue
        if difference[index] == 0:
            return float(x[index])
        if difference[index] * difference[index + 1] < 0:
            fraction = -difference[index] / (difference[index + 1] - difference[index])
            crossing = x[index] + fraction * (x[index + 1] - x[index])
            if interval[0] <= crossing <= interval[1]:
                return float(crossing)
    raise ValueError("curves do not cross inside the requested interval")

