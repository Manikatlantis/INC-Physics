"""Square-lattice Ising model utilities.

The Hamiltonian counts every nearest-neighbour bond once by summing the right
and down neighbours of every lattice site.  Periodic boundaries are used in
both directions and J=1 throughout the project.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


SpinArray = NDArray[np.integer]


def validate_spins(spins: SpinArray) -> None:
    """Raise ``ValueError`` unless ``spins`` is one or more square lattices."""

    array = np.asarray(spins)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        raise ValueError("spins must end in two equal lattice dimensions")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("all assigned spins must be either -1 or +1")


def energy(spins: SpinArray) -> NDArray[np.integer] | np.integer:
    """Return E=-sum_<ij> s_i s_j for one lattice or a batch of lattices."""

    array = np.asarray(spins)
    validate_spins(array)
    bonds = array * (
        np.roll(array, shift=-1, axis=-1)
        + np.roll(array, shift=-1, axis=-2)
    )
    return -np.sum(bonds, axis=(-2, -1), dtype=np.int64)


def magnetization(spins: SpinArray) -> NDArray[np.integer] | np.integer:
    """Return total (not per-site) magnetization for each lattice."""

    array = np.asarray(spins)
    validate_spins(array)
    return np.sum(array, axis=(-2, -1), dtype=np.int64)


def neighbor_indices(row: int, col: int, lattice_size: int) -> tuple[tuple[int, int], ...]:
    """Return the four periodic nearest-neighbour indices of a site."""

    if lattice_size < 2:
        raise ValueError("lattice_size must be at least 2")
    if not (0 <= row < lattice_size and 0 <= col < lattice_size):
        raise IndexError("site lies outside the lattice")
    return (
        ((row - 1) % lattice_size, col),
        ((row + 1) % lattice_size, col),
        (row, (col - 1) % lattice_size),
        (row, (col + 1) % lattice_size),
    )


def flip_energy_change(spins: SpinArray, row: int, col: int) -> int:
    """Return the energy change caused by flipping one spin."""

    array = np.asarray(spins)
    validate_spins(array)
    if array.ndim != 2:
        raise ValueError("flip_energy_change expects one lattice")
    size = array.shape[0]
    neighbour_sum = sum(int(array[index]) for index in neighbor_indices(row, col, size))
    return 2 * int(array[row, col]) * neighbour_sum

