"""Small CPU GFlowNet policies for raster-order Ising construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


@dataclass(frozen=True)
class ConditionedTrainingConfig:
    lattice_size: int
    seed: int
    steps: int
    batch_size: int
    hidden_sizes: tuple[int, ...]
    temperature_min: float = 1.5
    temperature_max: float = 3.2
    log_z_hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 2.0e-3
    final_learning_rate: float = 2.0e-4
    uniform_fraction: float = 0.35
    policy_fraction: float = 0.35
    structured_fraction: float = 0.30
    anchor_temperatures: tuple[float, ...] = (1.8, 2.269185314213022, 3.0)
    anchor_fraction: float = 0.50
    exploration_epsilon: float = 0.05
    gradient_clip_norm: float = 20.0
    log_every: int = 100
    cpu_threads: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MaskedLinear(nn.Linear):
    """Linear layer whose fixed binary mask enforces autoregressive causality."""

    def __init__(self, in_features: int, out_features: int, mask: Tensor) -> None:
        super().__init__(in_features, out_features)
        if mask.shape != self.weight.shape:
            raise ValueError("mask shape must match linear weight shape")
        self.register_buffer("mask", mask.to(dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight * self.mask, self.bias)


class MaskedAutoregressivePolicy(nn.Module):
    """MADE-style binary policy conditioned on inverse temperature."""

    def __init__(self, n_sites: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes or len(hidden_sizes) > 3 or max(hidden_sizes) > 256:
            raise ValueError("use one to three hidden layers of at most 256 units")
        self.n_sites = int(n_sites)

        # Spin input i and its beta interaction have degree i+1; beta itself has
        # degree zero and may affect every conditional. Output k has degree k+1
        # and may only depend on spins < k.
        spin_degrees = torch.arange(1, n_sites + 1)
        feature_degrees = torch.cat(
            (spin_degrees, spin_degrees.clone(), torch.zeros(1, dtype=torch.int64))
        )
        degrees: list[Tensor] = [
            feature_degrees
        ]
        for layer_index, size in enumerate(hidden_sizes):
            offset = layer_index * max(1, n_sites // max(1, len(hidden_sizes)))
            degrees.append((torch.arange(size) + offset) % n_sites)

        layers: list[nn.Module] = []
        for input_degrees, output_degrees in zip(degrees[:-1], degrees[1:], strict=True):
            mask = (input_degrees[None, :] <= output_degrees[:, None]).to(torch.float32)
            layers.extend(
                (MaskedLinear(input_degrees.numel(), output_degrees.numel(), mask), nn.SiLU())
            )
        output_degrees = torch.arange(1, n_sites + 1)
        final_mask = (degrees[-1][None, :] < output_degrees[:, None]).to(torch.float32)
        output = MaskedLinear(degrees[-1].numel(), n_sites, final_mask)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        direct_mask = (feature_degrees[None, :] < output_degrees[:, None]).to(torch.float32)
        self.direct = MaskedLinear(feature_degrees.numel(), n_sites, direct_mask)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)

    def forward(self, spins: Tensor, normalized_beta: Tensor) -> Tensor:
        beta_column = normalized_beta.unsqueeze(1)
        features = torch.cat((spins, spins * beta_column, beta_column), dim=1)
        flipped_features = torch.cat((-spins, -spins * beta_column, beta_column), dim=1)
        # Odd symmetrization enforces l_k(-prefix,beta)=-l_k(prefix,beta), hence
        # q(x|beta)=q(-x|beta) exactly for the zero-field Ising target.
        raw = self.network(features) + self.direct(features)
        flipped_raw = self.network(flipped_features) + self.direct(flipped_features)
        return 0.5 * (raw - flipped_raw)


class TemperatureConditionedGFlowNet(nn.Module):
    """Temperature-conditioned raster policy and learned log Z(beta) network."""

    def __init__(
        self,
        lattice_size: int,
        hidden_sizes: Sequence[int],
        log_z_hidden_sizes: Sequence[int] = (64, 64),
        temperature_min: float = 1.5,
        temperature_max: float = 3.2,
    ) -> None:
        super().__init__()
        if lattice_size < 2:
            raise ValueError("lattice_size must be at least 2")
        if not (0 < temperature_min < temperature_max):
            raise ValueError("temperature bounds must be positive and ordered")
        if not log_z_hidden_sizes or len(log_z_hidden_sizes) > 3:
            raise ValueError("log Z network needs one to three hidden layers")
        if max(log_z_hidden_sizes) > 256:
            raise ValueError("log Z hidden layers may have at most 256 units")

        self.lattice_size = int(lattice_size)
        self.n_sites = self.lattice_size**2
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.log_z_hidden_sizes = tuple(int(size) for size in log_z_hidden_sizes)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.beta_min = 1.0 / self.temperature_max
        self.beta_max = 1.0 / self.temperature_min

        self.policy = MaskedAutoregressivePolicy(self.n_sites, self.hidden_sizes)
        log_z_layers: list[nn.Module] = []
        input_size = 1
        for hidden_size in self.log_z_hidden_sizes:
            log_z_layers.extend((nn.Linear(input_size, hidden_size), nn.SiLU()))
            input_size = hidden_size
        log_z_output = nn.Linear(input_size, 1)
        log_z_layers.append(log_z_output)
        self.log_z_network = nn.Sequential(*log_z_layers)
        nn.init.zeros_(log_z_output.weight)
        nn.init.constant_(log_z_output.bias, self.n_sites * log(2.0))

    def normalized_beta(self, temperatures: Tensor) -> Tensor:
        if temperatures.ndim != 1:
            raise ValueError("temperatures must have shape (batch,)")
        if torch.any(temperatures <= 0):
            raise ValueError("temperatures must be positive")
        beta = temperatures.reciprocal()
        return 2.0 * (beta - self.beta_min) / (self.beta_max - self.beta_min) - 1.0

    def policy_logits(self, states: Tensor, temperatures: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_sites:
            raise ValueError("states must have shape (batch, n_sites)")
        if temperatures.shape != (states.shape[0],):
            raise ValueError("one temperature is required per state")
        return self.policy(states, self.normalized_beta(temperatures))

    def log_z(self, temperatures: Tensor) -> Tensor:
        normalized_beta = self.normalized_beta(temperatures)
        return self.log_z_network(normalized_beta.unsqueeze(1)).squeeze(1)

    def trajectory_log_prob(self, terminal_spins: Tensor, temperatures: Tensor) -> Tensor:
        if terminal_spins.ndim != 2 or terminal_spins.shape[1] != self.n_sites:
            raise ValueError("terminal_spins must have shape (batch, n_sites)")
        if temperatures.shape != (terminal_spins.shape[0],):
            raise ValueError("one temperature is required per terminal")
        logits = self.policy_logits(terminal_spins, temperatures)
        actions = ((terminal_spins + 1.0) / 2.0)
        return -F.binary_cross_entropy_with_logits(logits, actions, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample_terminal(
        self,
        temperatures: Tensor,
        generator: torch.Generator,
        exploration_epsilon: float = 0.0,
    ) -> Tensor:
        if temperatures.ndim != 1 or temperatures.numel() == 0:
            raise ValueError("temperatures must be a non-empty vector")
        if not (0.0 <= exploration_epsilon <= 1.0):
            raise ValueError("exploration_epsilon must lie in [0, 1]")
        states = torch.zeros(
            (temperatures.shape[0], self.n_sites),
            dtype=torch.float32,
            device=temperatures.device,
        )
        for step in range(self.n_sites):
            logits = self.policy_logits(states, temperatures)[:, step]
            probability_plus = torch.sigmoid(logits)
            if exploration_epsilon:
                probability_plus = (
                    (1.0 - exploration_epsilon) * probability_plus
                    + exploration_epsilon * 0.5
                )
            actions = torch.bernoulli(probability_plus, generator=generator)
            states[:, step] = actions * 2.0 - 1.0
        return states


def _conditioned_training_batch(
    model: TemperatureConditionedGFlowNet,
    config: ConditionedTrainingConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    half_batch = config.batch_size // 2
    fractions = np.array(
        [config.uniform_fraction, config.policy_fraction, config.structured_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or not np.isclose(float(fractions.sum()), 1.0):
        raise ValueError("conditioned training fractions must be non-negative and sum to one")
    counts = np.floor(fractions * half_batch).astype(int)
    counts[-1] += half_batch - int(counts.sum())

    def sample_temperatures(count: int) -> Tensor:
        temperatures = config.temperature_min + (
            config.temperature_max - config.temperature_min
        ) * torch.rand(count, generator=generator)
        anchor_count = int(round(count * config.anchor_fraction))
        if anchor_count and config.anchor_temperatures:
            anchors = torch.tensor(config.anchor_temperatures, dtype=torch.float32)
            if torch.any(anchors < config.temperature_min) or torch.any(
                anchors > config.temperature_max
            ):
                raise ValueError("anchor temperatures must lie inside the training interval")
            anchor_indices = torch.randint(
                0, anchors.numel(), (anchor_count,), generator=generator
            )
            temperatures[:anchor_count] = anchors[anchor_indices]
        return temperatures

    terminal_pieces: list[Tensor] = []
    temperature_pieces: list[Tensor] = []
    for source, count in zip(("uniform", "policy", "structured"), counts, strict=True):
        if count == 0:
            continue
        temperatures = sample_temperatures(int(count))
        if source == "uniform":
            actions = torch.randint(
                0,
                2,
                (count, model.n_sites),
                generator=generator,
            )
            terminals = actions.to(torch.float32) * 2.0 - 1.0
        elif source == "policy":
            terminals = model.sample_terminal(
                temperatures,
                generator,
                exploration_epsilon=config.exploration_epsilon,
            )
        else:
            scaled_temperature = (
                (temperatures - config.temperature_min)
                / (config.temperature_max - config.temperature_min)
            )
            # Noisy ordered states cover the dominant low-energy sector. At
            # higher T, block-domain states add low magnetization without
            # discarding the local correlations absent from uniform noise.
            flip_probability = 0.01 + 0.20 * scaled_temperature.pow(1.5)
            signs = torch.randint(0, 2, (count, 1), generator=generator).to(torch.float32)
            signs = signs * 2.0 - 1.0
            flips = torch.bernoulli(
                flip_probability[:, None].expand(-1, model.n_sites),
                generator=generator,
            )
            terminals = signs * (1.0 - 2.0 * flips)
            if model.lattice_size % 2 == 0:
                domain_probability = scaled_temperature.pow(3)
                use_domains = torch.rand(count, generator=generator) < domain_probability
                domain_count = int(torch.count_nonzero(use_domains))
                if domain_count:
                    coarse_size = model.lattice_size // 2
                    coarse = torch.randint(
                        0,
                        2,
                        (domain_count, coarse_size, coarse_size),
                        generator=generator,
                    ).to(torch.float32)
                    domains = (coarse * 2.0 - 1.0).repeat_interleave(2, 1).repeat_interleave(2, 2)
                    domain_noise_probability = (
                        0.05 * scaled_temperature[use_domains]
                    )[:, None]
                    domain_flips = torch.bernoulli(
                        domain_noise_probability.expand(-1, model.n_sites),
                        generator=generator,
                    )
                    domains = domains.reshape(domain_count, model.n_sites)
                    terminals[use_domains] = domains * (1.0 - 2.0 * domain_flips)
        terminal_pieces.append(terminals)
        temperature_pieces.append(temperatures)

    base_terminals = torch.cat(terminal_pieces, dim=0)
    base_temperatures = torch.cat(temperature_pieces, dim=0)
    terminals = torch.cat((base_terminals, -base_terminals), dim=0)
    temperatures = torch.cat((base_temperatures, base_temperatures), dim=0)
    if terminals.shape[0] < config.batch_size:
        missing = config.batch_size - terminals.shape[0]
        terminals = torch.cat((terminals, terminals[:missing]), dim=0)
        temperatures = torch.cat((temperatures, temperatures[:missing]), dim=0)
    permutation = torch.randperm(terminals.shape[0], generator=generator)
    selected = permutation[: config.batch_size]
    return terminals[selected], temperatures[selected]


def train_temperature_conditioned(
    config: ConditionedTrainingConfig,
) -> tuple[TemperatureConditionedGFlowNet, list[dict[str, float]]]:
    """Train a single beta-conditioned policy with trajectory balance."""

    if config.steps <= 0 or config.batch_size < 4:
        raise ValueError("steps and batch size must be positive")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model = TemperatureConditionedGFlowNet(
        config.lattice_size,
        config.hidden_sizes,
        config.log_z_hidden_sizes,
        config.temperature_min,
        config.temperature_max,
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
        terminals, temperatures = _conditioned_training_batch(model, config, generator)
        trajectory_log_prob = model.trajectory_log_prob(terminals, temperatures)
        log_z = model.log_z(temperatures)
        with torch.no_grad():
            log_reward = -torch_ising_energy(terminals, config.lattice_size) / temperatures
        residual = log_z + trajectory_log_prob - log_reward
        loss = residual.square().mean()
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
                    "log_z_mean": float(log_z.detach().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    model.eval()
    return model, history


@torch.no_grad()
def enumerate_conditioned_log_probs(
    model: TemperatureConditionedGFlowNet,
    terminal_spins: np.ndarray,
    temperature: float,
    batch_size: int = 4_096,
) -> np.ndarray:
    flattened = np.asarray(terminal_spins, dtype=np.float32).reshape(-1, model.n_sites)
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, flattened.shape[0], batch_size):
        batch = torch.from_numpy(flattened[start : start + batch_size])
        temperatures = torch.full((batch.shape[0],), temperature, dtype=torch.float32)
        outputs.append(
            model.trajectory_log_prob(batch, temperatures).cpu().numpy().astype(np.float64)
        )
    return np.concatenate(outputs)
