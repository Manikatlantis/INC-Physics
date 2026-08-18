from __future__ import annotations

import unittest

import numpy as np

from src.critical import heat_capacity_from_log_z, linear_crossing, local_quadratic_peak


class CriticalEstimatorTests(unittest.TestCase):
    def test_log_z_second_derivative(self) -> None:
        beta = np.linspace(0.3, 0.7, 40)
        temperatures = 1.0 / beta
        log_z = 3.0 + 2.0 * beta + 4.0 * beta**2
        result = heat_capacity_from_log_z(
            temperatures,
            log_z,
            n_sites=2,
            polynomial_degree=2,
            peak_interval=(1.5, 3.0),
        )
        expected = 4.0 * (1.0 / result.temperatures) ** 2
        np.testing.assert_allclose(result.heat_capacity_per_site, expected, atol=1e-10)

    def test_local_quadratic_peak(self) -> None:
        x = np.linspace(1.0, 4.0, 13)
        y = 5.0 - (x - 2.35) ** 2
        self.assertAlmostEqual(local_quadratic_peak(x, y), 2.35, places=12)

    def test_linear_crossing(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        first = np.array([0.0, 1.0, 2.0])
        second = np.array([2.0, 0.5, 0.0])
        self.assertAlmostEqual(linear_crossing(x, first, second, (1.0, 3.0)), 1.8)


if __name__ == "__main__":
    unittest.main()

