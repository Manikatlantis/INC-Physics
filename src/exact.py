"""Full-enumeration oracle for the L=4 Ising model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.ising import energy, magnetization
from src.observables import Observables, compute_observables


@dataclass(frozen=True)
class ExactResult:
    temperature: float
    log_partition: float
    observables: Observables


class ExactIsing:
    """Enumerate all 2^(L^2) states; intentionally restricted to L<=4."""

    def __init__(self, lattice_size: int = 4) -> None:
        if lattice_size < 2 or lattice_size > 4:
            raise ValueError("exact enumeration is restricted to 2 <= L <= 4")
        self.lattice_size = lattice_size
        self.n_sites = lattice_size**2
        state_ids = np.arange(1 << self.n_sites, dtype=np.uint64)
        bit_positions = np.arange(self.n_sites, dtype=np.uint64)
        bits = ((state_ids[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        self.spins: NDArray[np.int8] = (2 * bits - 1).reshape(
            -1, lattice_size, lattice_size
        )
        self.energies = np.asarray(energy(self.spins), dtype=np.int16)
        self.magnetizations = np.asarray(magnetization(self.spins), dtype=np.int16)

    def probabilities(self, temperature: float) -> tuple[NDArray[np.float64], float]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        log_weights = -self.energies.astype(np.float64) / temperature
        maximum = float(np.max(log_weights))
        shifted_sum = float(np.sum(np.exp(log_weights - maximum), dtype=np.float64))
        log_partition = maximum + np.log(shifted_sum)
        probabilities = np.exp(log_weights - log_partition)
        return probabilities, float(log_partition)

    def evaluate(self, temperature: float) -> ExactResult:
        probabilities, log_partition = self.probabilities(temperature)
        observables = compute_observables(
            self.energies,
            self.magnetizations,
            temperature,
            self.n_sites,
            weights=probabilities,
        )
        return ExactResult(float(temperature), log_partition, observables)

