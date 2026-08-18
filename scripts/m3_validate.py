#!/usr/bin/env python3
"""Validate M3 temperature-conditioned GFlowNets for L=4 and L=8."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inc-physics-matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.exact import ExactIsing  # noqa: E402
from src.gflownet import (  # noqa: E402
    ConditionedTrainingConfig,
    TemperatureConditionedGFlowNet,
    enumerate_conditioned_log_probs,
    train_temperature_conditioned,
)
from src.ising import energy, magnetization  # noqa: E402
from src.metropolis import MetropolisConfig, run_metropolis  # noqa: E402
from src.observables import compute_observables  # noqa: E402


RESULTS_DIR = ROOT / "results"
EXACT_TC = 2.0 / np.log1p(np.sqrt(2.0))
TEST_TEMPERATURES = (1.8, float(EXACT_TC), 3.0)
LOG_Z_TEMPERATURES = (
    1.5,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.25,
    float(EXACT_TC),
    2.3,
    2.4,
    2.5,
    2.6,
    2.8,
    3.0,
    3.2,
)
MODEL_SAMPLE_COUNT = 200_000
MODEL_SAMPLE_BATCH = 5_000
MODEL_SAMPLE_SEEDS = {1.8: 10_801, float(EXACT_TC): 10_802, 3.0: 10_803}
MCMC_SEEDS = {1.8: 11_801, float(EXACT_TC): 11_802, 3.0: 11_803}
L4_CONFIG = ConditionedTrainingConfig(
    lattice_size=4,
    seed=3_304,
    steps=4_000,
    batch_size=1_024,
    hidden_sizes=(128, 128),
    log_every=100,
)
L8_CONFIG = ConditionedTrainingConfig(
    lattice_size=8,
    seed=3_708,
    steps=16_000,
    batch_size=512,
    hidden_sizes=(256, 256),
    uniform_fraction=0.20,
    policy_fraction=0.60,
    structured_fraction=0.20,
    anchor_temperatures=(1.8, 1.8, float(EXACT_TC), 3.0),
    anchor_fraction=0.70,
    log_every=200,
)
MCMC_SETTINGS = {
    "n_chains": 128,
    "burn_sweeps": 2_000,
    "n_samples_per_chain": 2_000,
    "thin_sweeps": 2,
    "initialization": "mixed",
}


def git_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip() if process.returncode == 0 else "unavailable"

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def run_unit_tests() -> tuple[bool, dict[str, int]]:
    suite = unittest.TestLoader().discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return result.wasSuccessful(), {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }


def relative_error(measured: float, reference: float) -> float:
    if reference == 0:
        return abs(measured - reference)
    return abs(measured - reference) / abs(reference)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: TemperatureConditionedGFlowNet,
    config: ConditionedTrainingConfig,
    history: list[dict[str, float]],
    path: Path,
) -> dict[str, str]:
    torch.save(
        {
            "milestone": "M3",
            "model_class": "TemperatureConditionedGFlowNet",
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "training_history": history,
            "git": git_provenance(),
        },
        path,
    )
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


@torch.no_grad()
def sample_model(
    model: TemperatureConditionedGFlowNet,
    temperature: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = np.empty((sample_count, model.n_sites), dtype=np.int8)
    for start in range(0, sample_count, MODEL_SAMPLE_BATCH):
        count = min(MODEL_SAMPLE_BATCH, sample_count - start)
        temperatures = torch.full((count,), temperature, dtype=torch.float32)
        batch = model.sample_terminal(temperatures, generator)
        samples[start : start + count] = batch.cpu().numpy().astype(np.int8)
    return samples


def make_figure(payload: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    l4_tests = payload["l4"]["test_temperatures"]
    test_temperatures = np.array([record["temperature"] for record in l4_tests])
    kl_values = [record["kl_exact_to_model"] for record in l4_tests]
    axes[0, 0].plot(test_temperatures, kl_values, marker="o", label="conditioned L=4")
    axes[0, 0].axhline(0.08, color="black", linestyle=":", label="acceptance limit")
    axes[0, 0].set_ylabel(r"$D_{KL}(p_{exact}\Vert p_{GFN})$ (nats)")
    axes[0, 0].legend(frameon=False)

    log_z_grid = payload["l4"]["log_z_grid"]
    log_z_temperatures = np.array([record["temperature"] for record in log_z_grid])
    axes[0, 1].plot(
        log_z_temperatures,
        [record["exact_log_z"] for record in log_z_grid],
        color="black",
        linewidth=2,
        label="exact L=4",
    )
    axes[0, 1].plot(
        log_z_temperatures,
        [record["learned_log_z"] for record in log_z_grid],
        marker="o",
        linestyle="--",
        label="conditioned GFlowNet",
    )
    axes[0, 1].set_ylabel(r"$\ln Z(T)$")
    axes[0, 1].legend(frameon=False)

    l8_records = payload["l8"]["test_temperatures"]
    l8_temperatures = np.array([record["temperature"] for record in l8_records])
    observable_axes = [
        (axes[0, 2], "energy_per_site", r"energy/site $\langle E\rangle/N$"),
        (axes[1, 0], "abs_magnetization", r"$\langle|M|\rangle/N$"),
        (axes[1, 1], "susceptibility", r"susceptibility $\chi$"),
    ]
    for axis, field, label in observable_axes:
        axis.plot(
            l8_temperatures,
            [record["metropolis"][field] for record in l8_records],
            marker="o",
            linewidth=2,
            label="Metropolis L=8",
        )
        axis.plot(
            l8_temperatures,
            [record["gflownet"][field] for record in l8_records],
            marker="x",
            linestyle="--",
            linewidth=2,
            label="conditioned GFlowNet",
        )
        axis.set_ylabel(label)
        axis.legend(frameon=False)

    axes[1, 2].plot(
        l8_temperatures,
        [record["mode_fraction_m_gt_0"] for record in l8_records],
        marker="o",
        color="tab:purple",
    )
    axes[1, 2].axhline(0.5, color="black", linestyle=":", label="symmetric target")
    axes[1, 2].fill_between(
        [1.5, 3.2], 0.45, 0.55, color="tab:green", alpha=0.12, label="T=1.8 acceptance band"
    )
    axes[1, 2].set_ylim(0.4, 0.6)
    axes[1, 2].set_ylabel(r"fraction with $m>0$")
    axes[1, 2].legend(frameon=False)

    for axis in axes.ravel():
        axis.axvline(EXACT_TC, color="tab:red", linestyle="--", linewidth=1.2)
        axis.set_xlabel("temperature T")
        axis.grid(alpha=0.25)
    figure.suptitle("M3: one temperature-conditioned policy per lattice size")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"m3_metrics_{stamp}.json"
    figure_path = RESULTS_DIR / f"m3_conditioned_summary_{stamp}.png"

    print("M3 validation: complete unit-test suite")
    unit_tests_passed, unit_test_metrics = run_unit_tests()
    oracle = ExactIsing(4)

    print("M3 validation: training the single conditioned L=4 model", flush=True)
    l4_model, l4_history = train_temperature_conditioned(L4_CONFIG)
    l4_checkpoint = save_checkpoint(
        l4_model,
        L4_CONFIG,
        l4_history,
        RESULTS_DIR / f"m3_conditioned_L4_{stamp}.pt",
    )
    l4_test_records: list[dict[str, Any]] = []
    for temperature in TEST_TEMPERATURES:
        exact_probabilities, _ = oracle.probabilities(temperature)
        log_model_probabilities = enumerate_conditioned_log_probs(
            l4_model, oracle.spins, temperature
        )
        normalization = float(np.exp(log_model_probabilities).sum())
        kl = float(
            np.dot(
                exact_probabilities,
                np.log(exact_probabilities) - log_model_probabilities,
            )
        )
        l4_test_records.append(
            {
                "temperature": temperature,
                "kl_exact_to_model": kl,
                "probability_normalization": normalization,
                "passed": kl < 0.08 and abs(normalization - 1.0) < 2e-6,
            }
        )
        print(f"  L=4 T={temperature:.6f} exact KL={kl:.6f}", flush=True)

    l4_log_z_records: list[dict[str, float | bool]] = []
    for temperature in LOG_Z_TEMPERATURES:
        exact_log_z = oracle.evaluate(temperature).log_partition
        with torch.no_grad():
            learned_log_z = float(l4_model.log_z(torch.tensor([temperature]))[0])
        error = relative_error(learned_log_z, exact_log_z)
        l4_log_z_records.append(
            {
                "temperature": temperature,
                "exact_log_z": exact_log_z,
                "learned_log_z": learned_log_z,
                "relative_error": error,
                "passed": error < 0.03,
            }
        )
    maximum_log_z_error = max(float(record["relative_error"]) for record in l4_log_z_records)
    print(f"  L=4 maximum ln Z relative error={maximum_log_z_error:.3%}", flush=True)

    print("M3 validation: training the single conditioned L=8 model", flush=True)
    l8_model, l8_history = train_temperature_conditioned(L8_CONFIG)
    l8_checkpoint = save_checkpoint(
        l8_model,
        L8_CONFIG,
        l8_history,
        RESULTS_DIR / f"m3_conditioned_L8_{stamp}.pt",
    )
    l8_records: list[dict[str, Any]] = []
    for temperature in TEST_TEMPERATURES:
        model_seed = MODEL_SAMPLE_SEEDS[temperature]
        generated = sample_model(
            l8_model, temperature, MODEL_SAMPLE_COUNT, model_seed
        ).reshape(-1, 8, 8)
        generated_energies = np.asarray(energy(generated))
        generated_magnetizations = np.asarray(magnetization(generated))
        gfn_observables = compute_observables(
            generated_energies,
            generated_magnetizations,
            temperature,
            64,
        )

        mcmc_config = MetropolisConfig(
            lattice_size=8,
            temperature=temperature,
            seed=MCMC_SEEDS[temperature],
            **MCMC_SETTINGS,
        )
        mcmc_result = run_metropolis(mcmc_config)
        mcmc_observables = mcmc_result.observables
        errors = {
            "energy_per_site": relative_error(
                gfn_observables.energy_per_site, mcmc_observables.energy_per_site
            ),
            "abs_magnetization": relative_error(
                gfn_observables.abs_magnetization, mcmc_observables.abs_magnetization
            ),
            "susceptibility": relative_error(
                gfn_observables.susceptibility, mcmc_observables.susceptibility
            ),
        }
        mode_fraction = float(np.mean(generated_magnetizations > 0))
        criteria = {
            "energy_within_3_percent": errors["energy_per_site"] < 0.03,
            "abs_magnetization_within_3_percent": errors["abs_magnetization"] < 0.03,
            "susceptibility_within_10_percent": errors["susceptibility"] < 0.10,
        }
        record = {
            "temperature": temperature,
            "model_seed": model_seed,
            "model_sample_count": MODEL_SAMPLE_COUNT,
            "mcmc_seed": MCMC_SEEDS[temperature],
            "mcmc_sample_count": mcmc_result.n_samples,
            "mcmc_acceptance_rate": mcmc_result.acceptance_rate,
            "gflownet": gfn_observables.to_dict(),
            "metropolis": mcmc_observables.to_dict(),
            "relative_errors": errors,
            "mode_fraction_m_gt_0": mode_fraction,
            "criteria": criteria,
            "passed": bool(all(criteria.values())),
        }
        l8_records.append(record)
        print(
            f"  L=8 T={temperature:.6f} errors: E={errors['energy_per_site']:.3%}, "
            f"|m|={errors['abs_magnetization']:.3%}, chi={errors['susceptibility']:.3%}; "
            f"P(m>0)={mode_fraction:.4f} -> {'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )

    mode_fraction_t1p8 = next(
        record["mode_fraction_m_gt_0"]
        for record in l8_records
        if record["temperature"] == 1.8
    )
    mode_coverage_passed = 0.45 <= mode_fraction_t1p8 <= 0.55
    l4_kl_passed = all(record["passed"] for record in l4_test_records)
    l4_log_z_passed = all(bool(record["passed"]) for record in l4_log_z_records)
    l8_observables_passed = all(record["passed"] for record in l8_records)
    passed = (
        unit_tests_passed
        and l4_kl_passed
        and l4_log_z_passed
        and l8_observables_passed
        and mode_coverage_passed
    )
    finished = datetime.now().astimezone()
    payload: dict[str, Any] = {
        "milestone": "M3",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "git": git_provenance(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "device": "cpu",
        "exact_critical_temperature": float(EXACT_TC),
        "unit_tests": {"passed": unit_tests_passed, **unit_test_metrics},
        "acceptance_thresholds": {
            "l4_kl_nats_strictly_below": 0.08,
            "l4_log_z_relative_error_strictly_below": 0.03,
            "l8_energy_and_abs_m_relative_error_strictly_below": 0.03,
            "l8_susceptibility_relative_error_strictly_below": 0.10,
            "mode_fraction_interval_inclusive": [0.45, 0.55],
        },
        "l4": {
            "training_config": L4_CONFIG.to_dict(),
            "training_history": l4_history,
            "checkpoint": l4_checkpoint,
            "test_temperatures": l4_test_records,
            "log_z_grid": l4_log_z_records,
            "maximum_log_z_relative_error": maximum_log_z_error,
        },
        "l8": {
            "training_config": L8_CONFIG.to_dict(),
            "training_history": l8_history,
            "checkpoint": l8_checkpoint,
            "mcmc_settings": MCMC_SETTINGS,
            "test_temperatures": l8_records,
        },
        "criteria": {
            "a_l4_kl": l4_kl_passed,
            "b_l4_log_z": l4_log_z_passed,
            "c_l8_observables": l8_observables_passed,
            "d_low_temperature_mode_coverage": mode_coverage_passed,
            "unit_tests": unit_tests_passed,
        },
        "artifacts": {"figure": str(figure_path.relative_to(ROOT))},
    }
    make_figure(payload, figure_path)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"M3 unit tests: {unit_test_metrics['tests_run']} -> {'PASS' if unit_tests_passed else 'FAIL'}")
    print(f"M3 metrics: {metrics_path.relative_to(ROOT)}")
    print(f"M3 figure:  {figure_path.relative_to(ROOT)}")
    print(f"M3 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

