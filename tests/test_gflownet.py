from __future__ import annotations

import unittest

import numpy as np
import torch

from src.exact import ExactIsing
from src.gflownet import (
    FixedTemperatureGFlowNet,
    TemperatureConditionedGFlowNet,
    enumerate_conditioned_log_probs,
    enumerate_terminal_log_probs,
    torch_ising_energy,
)


class GFlowNetTests(unittest.TestCase):
    def test_torch_energy_matches_numpy_oracle(self) -> None:
        oracle = ExactIsing(2)
        flat = torch.from_numpy(oracle.spins.reshape(-1, 4).astype(np.float32))
        actual = torch_ising_energy(flat, 2).numpy()
        np.testing.assert_array_equal(actual, oracle.energies)

    def test_autoregressive_distribution_normalizes(self) -> None:
        torch.manual_seed(11)
        oracle = ExactIsing(2)
        model = FixedTemperatureGFlowNet(2, 2.0, hidden_sizes=(16,))
        log_probabilities = enumerate_terminal_log_probs(model, oracle.spins, batch_size=16)
        self.assertAlmostEqual(float(np.exp(log_probabilities).sum()), 1.0, places=6)

    def test_seeded_rollout_is_reproducible_and_terminal(self) -> None:
        model = FixedTemperatureGFlowNet(2, 2.0, hidden_sizes=(16,))
        first_generator = torch.Generator().manual_seed(23)
        second_generator = torch.Generator().manual_seed(23)
        first = model.sample_terminal(32, first_generator)
        second = model.sample_terminal(32, second_generator)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.all((first == -1) | (first == 1))))

    def test_uniform_policy_has_expected_trajectory_log_prob(self) -> None:
        model = FixedTemperatureGFlowNet(2, 3.0, hidden_sizes=(8,))
        terminals = torch.ones((3, 4), dtype=torch.float32)
        expected = -4.0 * np.log(2.0)
        np.testing.assert_allclose(
            model.trajectory_log_prob(terminals).detach().numpy(),
            expected,
            rtol=0,
            atol=1e-6,
        )

    def test_conditioned_mask_blocks_future_spins(self) -> None:
        torch.manual_seed(31)
        model = TemperatureConditionedGFlowNet(2, hidden_sizes=(16, 16))
        with torch.no_grad():
            model.policy.network[-1].weight.normal_(mean=0.0, std=0.2)
            model.policy.direct.weight.normal_(mean=0.0, std=0.2)
        first = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        changed_suffix = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
        temperatures = torch.tensor([2.2])
        first_logits = model.policy_logits(first, temperatures)
        changed_logits = model.policy_logits(changed_suffix, temperatures)
        # Output 2 may see sites 0 and 1, but not sites 2 and 3.
        self.assertAlmostEqual(
            float(first_logits[0, 2].detach()),
            float(changed_logits[0, 2].detach()),
            places=7,
        )

    def test_conditioned_policy_is_spin_flip_symmetric(self) -> None:
        torch.manual_seed(37)
        model = TemperatureConditionedGFlowNet(2, hidden_sizes=(16, 16))
        spins = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        temperatures = torch.tensor([1.8])
        logits = model.policy_logits(spins, temperatures)
        flipped_logits = model.policy_logits(-spins, temperatures)
        torch.testing.assert_close(logits, -flipped_logits)
        self.assertEqual(float(logits[0, 0].detach()), 0.0)

    def test_conditioned_distribution_normalizes(self) -> None:
        torch.manual_seed(41)
        oracle = ExactIsing(2)
        model = TemperatureConditionedGFlowNet(2, hidden_sizes=(16, 16))
        for temperature in (1.5, 2.269185314213022, 3.2):
            log_probabilities = enumerate_conditioned_log_probs(
                model, oracle.spins, temperature, batch_size=16
            )
            self.assertAlmostEqual(float(np.exp(log_probabilities).sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
