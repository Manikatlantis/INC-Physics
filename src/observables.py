"""Thermodynamic observables computed from energies and magnetizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Observables:
    """Finite-lattice observables at one temperature.

    Susceptibility uses the connected absolute-magnetization convention
    beta/N * (<M^2> - <|M|>^2).  This removes finite-volume tunnelling between
    the two symmetry-related ordered phases and exposes the critical peak.
    """

    temperature: float
    energy_per_site: float
    abs_magnetization: float
    susceptibility: float
    specific_heat: float
    binder_cumulant: float
    mean_energy: float
    mean_magnetization: float
    mean_abs_magnetization: float
    mean_energy_squared: float
    mean_magnetization_squared: float
    mean_magnetization_fourth: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64] | None) -> float:
    if weights is None:
        return float(np.mean(values, dtype=np.float64))
    return float(np.dot(weights, values))


def compute_observables(
    energies: ArrayLike,
    magnetizations: ArrayLike,
    temperature: float,
    n_sites: int,
    weights: ArrayLike | None = None,
) -> Observables:
    """Compute observables from exact weighted states or unweighted samples."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")

    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    magnetization_values = np.asarray(magnetizations, dtype=np.float64).reshape(-1)
    if energy_values.size == 0 or energy_values.shape != magnetization_values.shape:
        raise ValueError("energies and magnetizations must be non-empty and equally sized")

    normalized_weights: NDArray[np.float64] | None = None
    if weights is not None:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.shape != energy_values.shape:
            raise ValueError("weights must have the same shape as the samples")
        if np.any(normalized_weights < 0) or not np.all(np.isfinite(normalized_weights)):
            raise ValueError("weights must be finite and non-negative")
        total = float(np.sum(normalized_weights))
        if total <= 0:
            raise ValueError("weights must have positive total mass")
        normalized_weights = normalized_weights / total

    abs_magnetization_values = np.abs(magnetization_values)
    mean_e = _weighted_mean(energy_values, normalized_weights)
    mean_e2 = _weighted_mean(energy_values**2, normalized_weights)
    mean_m = _weighted_mean(magnetization_values, normalized_weights)
    mean_abs_m = _weighted_mean(abs_magnetization_values, normalized_weights)
    mean_m2 = _weighted_mean(magnetization_values**2, normalized_weights)
    mean_m4 = _weighted_mean(magnetization_values**4, normalized_weights)

    beta = 1.0 / temperature
    energy_variance = max(0.0, mean_e2 - mean_e**2)
    connected_magnetization_variance = max(0.0, mean_m2 - mean_abs_m**2)
    binder = 0.0 if mean_m2 == 0 else 1.0 - mean_m4 / (3.0 * mean_m2**2)

    return Observables(
        temperature=float(temperature),
        energy_per_site=mean_e / n_sites,
        abs_magnetization=mean_abs_m / n_sites,
        susceptibility=beta * connected_magnetization_variance / n_sites,
        specific_heat=beta**2 * energy_variance / n_sites,
        binder_cumulant=binder,
        mean_energy=mean_e,
        mean_magnetization=mean_m,
        mean_abs_magnetization=mean_abs_m,
        mean_energy_squared=mean_e2,
        mean_magnetization_squared=mean_m2,
        mean_magnetization_fourth=mean_m4,
    )

