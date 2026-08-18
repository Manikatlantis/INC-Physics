"""Seeded CPU Metropolis-Hastings sampler for even square lattices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.ising import energy, magnetization
from src.observables import Observables, compute_observables


@dataclass(frozen=True)
class MetropolisConfig:
    lattice_size: int
    temperature: float
    seed: int
    n_chains: int = 128
    burn_sweeps: int = 1_000
    n_samples_per_chain: int = 2_000
    thin_sweeps: int = 2
    initialization: str = "mixed"

    @property
    def n_sites(self) -> int:
        return self.lattice_size**2


@dataclass(frozen=True)
class MetropolisResult:
    config: MetropolisConfig
    energies: NDArray[np.int16]
    magnetizations: NDArray[np.int16]
    acceptance_rate: float
    observables: Observables

    @property
    def n_samples(self) -> int:
        return int(self.energies.size)


def _initial_spins(config: MetropolisConfig, rng: np.random.Generator) -> NDArray[np.int8]:
    shape = (config.n_chains, config.lattice_size, config.lattice_size)
    if config.initialization == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)
    if config.initialization == "ordered":
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(config.n_chains, 1, 1))
        return np.broadcast_to(signs, shape).copy()
    if config.initialization != "mixed":
        raise ValueError("initialization must be 'random', 'ordered', or 'mixed'")

    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)
    one_third = config.n_chains // 3
    spins[:one_third] = 1
    spins[one_third : 2 * one_third] = -1
    return spins


def run_metropolis(config: MetropolisConfig) -> MetropolisResult:
    """Run independent chains using exact checkerboard Metropolis sweeps.

    Sites of one parity do not share bonds on the supported even lattices, so
    all proposals on a sublattice can be evaluated in parallel without changing
    the Markov kernel.  One sweep proposes every lattice site once.
    """

    if config.lattice_size < 2 or config.lattice_size % 2 != 0:
        raise ValueError("checkerboard updates require a positive even lattice size")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if min(
        config.n_chains,
        config.n_samples_per_chain,
        config.thin_sweeps,
    ) <= 0 or config.burn_sweeps < 0:
        raise ValueError("chain counts and sampling intervals must be positive")

    rng = np.random.default_rng(config.seed)
    spins = _initial_spins(config, rng)
    indices = np.indices((config.lattice_size, config.lattice_size))
    parity_masks = tuple(((indices[0] + indices[1]) % 2 == parity)[None, :, :] for parity in (0, 1))
    beta = 1.0 / config.temperature
    accepted = 0
    proposed = 0

    def sweep() -> None:
        nonlocal accepted, proposed, spins
        for mask in parity_masks:
            neighbour_sum = (
                np.roll(spins, 1, axis=1)
                + np.roll(spins, -1, axis=1)
                + np.roll(spins, 1, axis=2)
                + np.roll(spins, -1, axis=2)
            )
            delta_energy = 2 * spins * neighbour_sum
            log_uniform = np.log(rng.random(spins.shape))
            accept = mask & ((delta_energy <= 0) | (log_uniform < -beta * delta_energy))
            spins[accept] *= -1
            accepted += int(np.count_nonzero(accept))
            proposed += config.n_chains * config.n_sites // 2

    for _ in range(config.burn_sweeps):
        sweep()

    sample_shape = (config.n_samples_per_chain, config.n_chains)
    sampled_energies = np.empty(sample_shape, dtype=np.int16)
    sampled_magnetizations = np.empty(sample_shape, dtype=np.int16)
    for sample_index in range(config.n_samples_per_chain):
        for _ in range(config.thin_sweeps):
            sweep()
        sampled_energies[sample_index] = energy(spins)
        sampled_magnetizations[sample_index] = magnetization(spins)

    flattened_energies = sampled_energies.reshape(-1)
    flattened_magnetizations = sampled_magnetizations.reshape(-1)
    observables = compute_observables(
        flattened_energies,
        flattened_magnetizations,
        config.temperature,
        config.n_sites,
    )
    return MetropolisResult(
        config=config,
        energies=flattened_energies,
        magnetizations=flattened_magnetizations,
        acceptance_rate=accepted / proposed,
        observables=observables,
    )

