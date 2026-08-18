#!/usr/bin/env python3
"""Validate M2 fixed-temperature trajectory-balance GFlowNets at L=4."""

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
    FixedTrainingConfig,
    enumerate_terminal_log_probs,
    train_fixed_temperature,
)
from src.observables import compute_observables  # noqa: E402


RESULTS_DIR = ROOT / "results"
TEMPERATURES = (3.0, 2.0)
TRAINING_SEEDS = {3.0: 230, 2.0: 220}
EMPIRICAL_SEEDS = {3.0: 3_230, 2.0: 3_220}
EMPIRICAL_SAMPLES = 2_000_000
JEFFREYS_PSEUDOCOUNT = 0.5
TRAINING_STEPS = 8_000
BATCH_SIZE = 1_024
HIDDEN_SIZES = (128, 128)


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


def energy_level_masses(
    energies: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, list[float]]:
    levels = np.unique(energies)
    masses = [float(np.sum(probabilities[energies == level])) for level in levels]
    return {"energies": levels.astype(float).tolist(), "probability_mass": masses}


def make_figure(records: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for column, record in enumerate(records):
        temperature = record["temperature"]
        history = record["training_history"]
        steps = [point["step"] for point in history]
        losses = [point["loss"] for point in history]
        axes[0, column].semilogy(steps, losses, color="tab:blue")
        axes[0, column].set_title(f"T={temperature:g}: trajectory-balance training")
        axes[0, column].set_xlabel("optimizer step")
        axes[0, column].set_ylabel("mean squared TB residual")
        axes[0, column].grid(alpha=0.25)

        exact_levels = record["energy_level_mass"]["exact"]
        empirical_levels = record["energy_level_mass"]["empirical"]
        axes[1, column].plot(
            exact_levels["energies"],
            exact_levels["probability_mass"],
            marker="o",
            linewidth=2,
            label="exact",
        )
        axes[1, column].plot(
            empirical_levels["energies"],
            empirical_levels["probability_mass"],
            marker="x",
            linestyle="--",
            label="GFlowNet empirical",
        )
        axes[1, column].set_xlabel("terminal Ising energy E")
        axes[1, column].set_ylabel("probability mass")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend(frameon=False)

    figure.suptitle("M2: fixed-temperature L=4 GFlowNets against exact enumeration")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"m2_metrics_{stamp}.json"
    figure_path = RESULTS_DIR / f"m2_fixed_temperature_{stamp}.png"

    print("M2 validation: complete unit-test suite")
    unit_tests_passed, unit_test_metrics = run_unit_tests()
    oracle = ExactIsing(4)
    flattened_spins = oracle.spins.reshape(-1, oracle.n_sites)
    records: list[dict[str, Any]] = []

    for temperature in TEMPERATURES:
        print(
            f"M2 validation: training L=4 T={temperature:g} for "
            f"{TRAINING_STEPS} trajectory-balance steps",
            flush=True,
        )
        config = FixedTrainingConfig(
            lattice_size=4,
            temperature=temperature,
            seed=TRAINING_SEEDS[temperature],
            steps=TRAINING_STEPS,
            batch_size=BATCH_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        )
        model, history = train_fixed_temperature(config)

        exact_probabilities, exact_log_z = oracle.probabilities(temperature)
        model_log_probabilities = enumerate_terminal_log_probs(model, oracle.spins)
        model_probabilities = np.exp(model_log_probabilities)
        normalization = float(np.sum(model_probabilities))
        if not np.isclose(normalization, 1.0, atol=2e-6):
            raise RuntimeError(f"autoregressive probabilities sum to {normalization}")
        model_probabilities /= normalization

        exact_log_probabilities = np.log(exact_probabilities)
        exact_to_model_kl = float(
            np.dot(exact_probabilities, exact_log_probabilities - np.log(model_probabilities))
        )

        empirical_rng = np.random.default_rng(EMPIRICAL_SEEDS[temperature])
        counts = empirical_rng.multinomial(EMPIRICAL_SAMPLES, model_probabilities)
        raw_empirical_probabilities = counts.astype(np.float64) / EMPIRICAL_SAMPLES
        empirical_probabilities = (counts + JEFFREYS_PSEUDOCOUNT) / (
            EMPIRICAL_SAMPLES + JEFFREYS_PSEUDOCOUNT * counts.size
        )
        empirical_kl = float(
            np.dot(exact_probabilities, exact_log_probabilities - np.log(empirical_probabilities))
        )
        unobserved = counts == 0
        unobserved_exact_mass = float(np.sum(exact_probabilities[unobserved]))

        empirical_observables = compute_observables(
            oracle.energies,
            oracle.magnetizations,
            temperature,
            oracle.n_sites,
            weights=raw_empirical_probabilities,
        )
        exact_observables = oracle.evaluate(temperature).observables
        energy_error = relative_error(
            empirical_observables.energy_per_site,
            exact_observables.energy_per_site,
        )
        magnetization_error = relative_error(
            empirical_observables.abs_magnetization,
            exact_observables.abs_magnetization,
        )
        learned_log_z = float(model.log_z.detach())
        log_z_error = relative_error(learned_log_z, exact_log_z)

        safe_temperature = str(temperature).replace(".", "p")
        checkpoint_path = RESULTS_DIR / f"m2_model_T{safe_temperature}_{stamp}.pt"
        torch.save(
            {
                "milestone": "M2",
                "model_class": "FixedTemperatureGFlowNet",
                "model_state_dict": model.state_dict(),
                "training_config": config.to_dict(),
                "training_history": history,
                "git": git_provenance(),
            },
            checkpoint_path,
        )

        record = {
            "temperature": temperature,
            "training_seed": TRAINING_SEEDS[temperature],
            "empirical_seed": EMPIRICAL_SEEDS[temperature],
            "training_config": config.to_dict(),
            "training_history": history,
            "terminal_probability_normalization": normalization,
            "model_kl_exact_to_gflownet": exact_to_model_kl,
            "empirical": {
                "sample_count": EMPIRICAL_SAMPLES,
                "observed_state_count": int(np.count_nonzero(counts)),
                "total_state_count": int(counts.size),
                "unobserved_exact_probability_mass": unobserved_exact_mass,
                "histogram_estimator": "Jeffreys additive pseudocount",
                "pseudocount_per_state": JEFFREYS_PSEUDOCOUNT,
                "kl_exact_to_empirical": empirical_kl,
            },
            "observables": {
                "exact": exact_observables.to_dict(),
                "empirical": empirical_observables.to_dict(),
                "relative_errors": {
                    "energy_per_site": energy_error,
                    "abs_magnetization": magnetization_error,
                },
            },
            "partition": {
                "exact_log_z": exact_log_z,
                "learned_log_z": learned_log_z,
                "relative_error": log_z_error,
            },
            "energy_level_mass": {
                "exact": energy_level_masses(oracle.energies, exact_probabilities),
                "empirical": energy_level_masses(oracle.energies, raw_empirical_probabilities),
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
            },
            "criteria": {
                "empirical_kl_below_0p05": empirical_kl < 0.05,
                "energy_within_2_percent": energy_error < 0.02,
                "abs_magnetization_within_2_percent": magnetization_error < 0.02,
                "log_z_within_2_percent": log_z_error < 0.02,
            },
        }
        record["passed"] = bool(all(record["criteria"].values()))
        records.append(record)
        print(
            f"  T={temperature:g} empirical KL={empirical_kl:.6f}, "
            f"model KL={exact_to_model_kl:.6f}, "
            f"E error={energy_error:.3%}, |m| error={magnetization_error:.3%}, "
            f"logZ error={log_z_error:.3%} -> "
            f"{'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )

    make_figure(records, figure_path)
    passed = unit_tests_passed and all(record["passed"] for record in records)
    finished = datetime.now().astimezone()
    payload = {
        "milestone": "M2",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "git": git_provenance(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "device": "cpu",
        "unit_tests": {"passed": unit_tests_passed, **unit_test_metrics},
        "acceptance_thresholds": {
            "empirical_kl_nats_strictly_below": 0.05,
            "observable_relative_error_strictly_below": 0.02,
            "log_z_relative_error_strictly_below": 0.02,
            "minimum_empirical_samples": 200_000,
        },
        "records": records,
        "artifacts": {"figure": str(figure_path.relative_to(ROOT))},
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"M2 unit tests: {unit_test_metrics['tests_run']} -> {'PASS' if unit_tests_passed else 'FAIL'}")
    print(f"M2 metrics: {metrics_path.relative_to(ROOT)}")
    print(f"M2 figure:  {figure_path.relative_to(ROOT)}")
    print(f"M2 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

