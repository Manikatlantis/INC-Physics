from __future__ import annotations

import unittest

import numpy as np

from src.exact import ExactIsing
from src.ising import energy, flip_energy_change, magnetization, neighbor_indices
from src.metropolis import MetropolisConfig, run_metropolis


class IsingEnergyTests(unittest.TestCase):
    def test_aligned_state_has_ground_energy(self) -> None:
        spins = np.ones((4, 4), dtype=np.int8)
        self.assertEqual(int(energy(spins)), -32)
        self.assertEqual(int(magnetization(spins)), 16)

    def test_checkerboard_has_maximum_energy(self) -> None:
        rows, columns = np.indices((4, 4))
        spins = np.where((rows + columns) % 2 == 0, 1, -1).astype(np.int8)
        self.assertEqual(int(energy(spins)), 32)
        self.assertEqual(int(magnetization(spins)), 0)

    def test_single_flip_cost_matches_full_energy_difference(self) -> None:
        spins = np.ones((4, 4), dtype=np.int8)
        before = int(energy(spins))
        predicted = flip_energy_change(spins, 0, 0)
        spins[0, 0] *= -1
        self.assertEqual(predicted, 8)
        self.assertEqual(int(energy(spins)) - before, predicted)

    def test_batch_energy(self) -> None:
        aligned = np.ones((4, 4), dtype=np.int8)
        checkerboard = np.fromfunction(
            lambda row, col: np.where((row + col) % 2 == 0, 1, -1),
            (4, 4),
            dtype=int,
        ).astype(np.int8)
        np.testing.assert_array_equal(energy(np.stack([aligned, checkerboard])), [-32, 32])


class PeriodicBoundaryTests(unittest.TestCase):
    def test_corner_neighbors_wrap_both_axes(self) -> None:
        self.assertEqual(
            set(neighbor_indices(0, 0, 4)),
            {(3, 0), (1, 0), (0, 3), (0, 1)},
        )

    def test_wrapped_bond_is_counted(self) -> None:
        spins = np.ones((4, 4), dtype=np.int8)
        spins[0, 0] = -1
        self.assertEqual(int(energy(spins)), -24)


class ExactEnumerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = ExactIsing(4)

    def test_enumerates_every_state(self) -> None:
        self.assertEqual(self.oracle.spins.shape, (65_536, 4, 4))

    def test_ground_state_degeneracy(self) -> None:
        self.assertEqual(int(np.count_nonzero(self.oracle.energies == -32)), 2)

    def test_probabilities_normalize(self) -> None:
        probabilities, log_partition = self.oracle.probabilities(2.269185314213022)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertTrue(np.isfinite(log_partition))


class SeedTests(unittest.TestCase):
    def test_metropolis_is_reproducible(self) -> None:
        config = MetropolisConfig(
            lattice_size=4,
            temperature=2.5,
            seed=7,
            n_chains=4,
            burn_sweeps=5,
            n_samples_per_chain=8,
            thin_sweeps=1,
        )
        first = run_metropolis(config)
        second = run_metropolis(config)
        np.testing.assert_array_equal(first.energies, second.energies)
        np.testing.assert_array_equal(first.magnetizations, second.magnetizations)


if __name__ == "__main__":
    unittest.main()

