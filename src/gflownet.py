"""Small CPU GFlowNet policies for raster-order Ising construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FixedTrainingConfig:
    lattice_size: int
    temperature: float
    seed: int
    steps: int = 8_000
    batch_size: int = 1_024
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 2.0e-3
    final_learning_rate: float = 2.0e-4
    uniform_fraction: float = 0.5
    exploration_epsilon: float = 0.05
    gradient_clip_norm: float = 10.0
    log_every: int = 100
    cpu_threads: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def torch_ising_energy(flat_spins: Tensor, lattice_size: int) -> Tensor:
    """Vectorized periodic Ising energy for flattened spin batches."""

    if flat_spins.ndim != 2 or flat_spins.shape[1] != lattice_size**2:
        raise ValueError("flat_spins must have shape (batch, lattice_size**2)")
    lattice = flat_spins.reshape(-1, lattice_size, lattice_size)
    bonds = lattice * (
        torch.roll(lattice, shifts=-1, dims=1)
        + torch.roll(lattice, shifts=-1, dims=2)
    )
    return -bonds.sum(dim=(1, 2))


class FixedTemperatureGFlowNet(nn.Module):
    """Autoregressive forward policy with a learned scalar log-partition.

    The state is a flattened lattice whose assigned prefix is in {-1,+1} and
    whose unassigned suffix is zero.  The only actions assign -1 or +1 to the
    next raster-order site.  There is one trajectory per terminal state and the
    reverse transition probability is one, making trajectory balance

        log Z + sum_t log P_F(a_t | s_t) = -E(x)/T.
    """

    def __init__(
        self,
        lattice_size: int,
        temperature: float,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if lattice_size < 2:
            raise ValueError("lattice_size must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not hidden_sizes or len(hidden_sizes) > 3:
            raise ValueError("use between one and three hidden layers")
        if max(hidden_sizes) > 256:
            raise ValueError("hidden layers may have at most 256 units")

        self.lattice_size = int(lattice_size)
        self.n_sites = self.lattice_size**2
        self.temperature = float(temperature)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        layers: list[nn.Module] = []
        input_size = self.n_sites + 1
        for hidden_size in self.hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.SiLU()))
            input_size = hidden_size
        output = nn.Linear(input_size, 2)
        layers.append(output)
        self.policy = nn.Sequential(*layers)
        self.log_z = nn.Parameter(torch.tensor(self.n_sites * log(2.0), dtype=torch.float32))

        # Begin from a uniform policy. This is exactly spin-flip symmetric and
        # avoids an arbitrary initial preference for either ordered phase.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def policy_logits(self, states: Tensor, steps: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_sites:
            raise ValueError("states must have shape (batch, n_sites)")
        if steps.ndim != 1 or steps.shape[0] != states.shape[0]:
            raise ValueError("steps must have shape (batch,)")
        step_feature = steps.to(dtype=states.dtype).unsqueeze(1) / self.n_sites
        return self.policy(torch.cat((states, step_feature), dim=1))

    def trajectory_log_prob(self, terminal_spins: Tensor) -> Tensor:
        """Return sum_t log P_F for supplied terminal trajectories."""

        if terminal_spins.ndim != 2 or terminal_spins.shape[1] != self.n_sites:
            raise ValueError("terminal_spins must have shape (batch, n_sites)")
        if not torch.all((terminal_spins == -1) | (terminal_spins == 1)):
            raise ValueError("terminal spins must be -1 or +1")

        batch_size = terminal_spins.shape[0]
        positions = torch.arange(self.n_sites, device=terminal_spins.device)
        prefix_mask = positions[None, None, :] < positions[None, :, None]
        prefix_states = terminal_spins[:, None, :] * prefix_mask.to(terminal_spins.dtype)
        repeated_steps = positions[None, :].expand(batch_size, -1)
        logits = self.policy_logits(
            prefix_states.reshape(batch_size * self.n_sites, self.n_sites),
            repeated_steps.reshape(-1),
        ).reshape(batch_size, self.n_sites, 2)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        actions = ((terminal_spins + 1) / 2).to(torch.int64)
        selected = torch.gather(log_probabilities, dim=2, index=actions.unsqueeze(-1))
        return selected.squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def sample_terminal(
        self,
        batch_size: int,
        generator: torch.Generator,
        exploration_epsilon: float = 0.0,
    ) -> Tensor:
        """Roll out forward trajectories with optional uniform exploration."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0.0 <= exploration_epsilon <= 1.0):
            raise ValueError("exploration_epsilon must lie in [0, 1]")
        device = self.log_z.device
        states = torch.zeros((batch_size, self.n_sites), dtype=torch.float32, device=device)
        for step in range(self.n_sites):
            steps = torch.full((batch_size,), step, dtype=torch.int64, device=device)
            probabilities = torch.softmax(self.policy_logits(states, steps), dim=-1)
            if exploration_epsilon:
                probabilities = (
                    (1.0 - exploration_epsilon) * probabilities
                    + exploration_epsilon * 0.5
                )
            actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            states[:, step] = actions.to(torch.float32) * 2.0 - 1.0
        return states


def _training_terminals(
    model: FixedTemperatureGFlowNet,
    config: FixedTrainingConfig,
    generator: torch.Generator,
) -> Tensor:
    """Build a full-support, exactly spin-flip-paired training batch."""

    half_batch = config.batch_size // 2
    uniform_count = int(round(half_batch * config.uniform_fraction))
    policy_count = half_batch - uniform_count
    pieces: list[Tensor] = []
    if uniform_count:
        uniform_actions = torch.randint(
            0,
            2,
            (uniform_count, model.n_sites),
            generator=generator,
            device=model.log_z.device,
        )
        pieces.append(uniform_actions.to(torch.float32) * 2.0 - 1.0)
    if policy_count:
        pieces.append(
            model.sample_terminal(
                policy_count,
                generator,
                exploration_epsilon=config.exploration_epsilon,
            )
        )
    base = torch.cat(pieces, dim=0)
    paired = torch.cat((base, -base), dim=0)
    if paired.shape[0] < config.batch_size:
        extra = paired[: config.batch_size - paired.shape[0]]
        paired = torch.cat((paired, extra), dim=0)
    permutation = torch.randperm(paired.shape[0], generator=generator, device=paired.device)
    return paired[permutation[: config.batch_size]]


def train_fixed_temperature(
    config: FixedTrainingConfig,
) -> tuple[FixedTemperatureGFlowNet, list[dict[str, float]]]:
    """Train with squared trajectory-balance loss on CPU."""

    if config.batch_size < 4 or not (0.0 <= config.uniform_fraction <= 1.0):
        raise ValueError("invalid training batch configuration")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    model = FixedTemperatureGFlowNet(
        config.lattice_size,
        config.temperature,
        config.hidden_sizes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )
    history: list[dict[str, float]] = []

    model.train()
    for step in range(1, config.steps + 1):
        terminals = _training_terminals(model, config, generator)
        trajectory_log_prob = model.trajectory_log_prob(terminals)
        with torch.no_grad():
            log_reward = -torch_ising_energy(terminals, config.lattice_size) / config.temperature
        residual = model.log_z + trajectory_log_prob - log_reward
        loss = torch.mean(residual.square())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        optimizer.step()
        scheduler.step()

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "residual_mean": float(residual.detach().mean()),
                    "residual_std": float(residual.detach().std()),
                    "log_z": float(model.log_z.detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    model.eval()
    return model, history


@torch.no_grad()
def enumerate_terminal_log_probs(
    model: FixedTemperatureGFlowNet,
    terminal_spins: np.ndarray,
    batch_size: int = 2_048,
) -> np.ndarray:
    """Evaluate normalized autoregressive probabilities for an enumerated oracle."""

    flattened = np.asarray(terminal_spins, dtype=np.float32).reshape(-1, model.n_sites)
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, flattened.shape[0], batch_size):
        batch = torch.from_numpy(flattened[start : start + batch_size])
        outputs.append(model.trajectory_log_prob(batch).cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)

