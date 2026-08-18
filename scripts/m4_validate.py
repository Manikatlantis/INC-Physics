#!/usr/bin/env python3
"""Validate M4: infer criticality from learned log Z and generated observables."""

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
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inc-physics-matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.critical import (  # noqa: E402
    HeatCapacityCurve,
    heat_capacity_from_log_z,
    linear_crossing,
    local_quadratic_peak,
)
from src.exact import ExactIsing  # noqa: E402
from src.gflownet import TemperatureConditionedGFlowNet  # noqa: E402
from src.ising import energy, magnetization  # noqa: E402
from src.metropolis import MetropolisConfig, run_metropolis  # noqa: E402
from src.observables import compute_observables  # noqa: E402


RESULTS_DIR = ROOT / "results"
EXACT_TC = 2.0 / np.log1p(np.sqrt(2.0))
TEMPERATURES = np.array(
    [
        1.50,
        1.70,
        1.90,
        2.00,
        2.10,
        2.15,
        2.20,
        2.25,
        float(EXACT_TC),
        2.30,
        2.35,
        2.40,
        2.45,
        2.50,
        2.55,
        2.60,
        2.70,
        2.80,
        3.00,
        3.20,
    ],
    dtype=np.float64,
)
LOG_Z_BETA_GRID = np.linspace(1.0 / 3.2, 1.0 / 1.5, 256, dtype=np.float64)
LOG_Z_POLYNOMIAL_DEGREE = 10
MODEL_SAMPLES_PER_TEMPERATURE = 100_000
MODEL_SAMPLE_BATCH = 5_000
MODEL_SAMPLE_BASE_SEEDS = {4: 14_400, 8: 14_800}
MCMC_BASE_SEED = 15_200
MCMC_SETTINGS = {
    "n_chains": 96,
    "burn_sweeps": 1_500,
    "n_samples_per_chain": 1_500,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_m3_metrics() -> Path:
    matches = sorted(RESULTS_DIR.glob("m3_metrics_*.json"))
    if not matches:
        raise FileNotFoundError("M3 metrics are required before M4")
    return matches[-1]


def load_conditioned_model(
    m3_payload: dict[str, Any], section: str
) -> tuple[TemperatureConditionedGFlowNet, dict[str, Any]]:
    checkpoint_record = m3_payload[section]["checkpoint"]
    checkpoint_path = ROOT / checkpoint_record["path"]
    if sha256(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["training_config"]
    model = TemperatureConditionedGFlowNet(
        config["lattice_size"],
        config["hidden_sizes"],
        config["log_z_hidden_sizes"],
        config["temperature_min"],
        config["temperature_max"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "path": str(checkpoint_path.relative_to(ROOT)),
        "sha256": checkpoint_record["sha256"],
        "training_config": config,
    }


@torch.no_grad()
def sample_with_entropy(
    model: TemperatureConditionedGFlowNet,
    temperature: float,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate terminal states and average Bernoulli entropy at every step."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = np.empty((sample_count, model.n_sites), dtype=np.int8)
    entropy_sum = np.zeros(model.n_sites, dtype=np.float64)
    for start in range(0, sample_count, MODEL_SAMPLE_BATCH):
        count = min(MODEL_SAMPLE_BATCH, sample_count - start)
        temperatures = torch.full((count,), temperature, dtype=torch.float32)
        states = torch.zeros((count, model.n_sites), dtype=torch.float32)
        for step in range(model.n_sites):
            logits = model.policy_logits(states, temperatures)[:, step]
            probability_plus = torch.sigmoid(logits)
            clipped = torch.clamp(probability_plus, 1e-8, 1.0 - 1e-8)
            entropy = -clipped * torch.log(clipped) - (1.0 - clipped) * torch.log(
                1.0 - clipped
            )
            entropy_sum[step] += float(torch.sum(entropy))
            actions = torch.bernoulli(probability_plus, generator=generator)
            states[:, step] = actions * 2.0 - 1.0
        samples[start : start + count] = states.numpy().astype(np.int8)
    return samples, entropy_sum / sample_count


def curve_record(curve: HeatCapacityCurve) -> dict[str, Any]:
    return {
        "temperatures": curve.temperatures.tolist(),
        "heat_capacity_per_site": curve.heat_capacity_per_site.tolist(),
        "peak_temperature": curve.peak_temperature,
        "peak_heat_capacity_per_site": curve.peak_heat_capacity_per_site,
        "fit_rms_log_z": curve.fit_rms_log_z,
        "polynomial_degree": curve.polynomial_degree,
    }


def make_summary_figure(payload: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    log_z_route = payload["log_z_route"]
    for key, label, color in (
        ("exact_l4_pipeline", "exact log Z, L=4", "black"),
        ("learned_l4", "learned log Z, L=4", "tab:blue"),
        ("learned_l8", "learned log Z, L=8", "tab:green"),
    ):
        curve = log_z_route[key]
        axes[0].plot(
            curve["temperatures"],
            curve["heat_capacity_per_site"],
            color=color,
            label=f"{label} (peak {curve['peak_temperature']:.3f})",
        )
    axes[0].set_ylabel("specific heat per site from log Z")
    axes[0].legend(frameon=False, fontsize=8)

    records = payload["observable_route"]["temperature_records"]
    temperatures = np.array([record["temperature"] for record in records])
    for size, label, style in (
        ("4", "GFlowNet L=4", "-"),
        ("8", "GFlowNet L=8", "--"),
        ("12", "Metropolis L=12", "-."),
    ):
        axes[1].plot(
            temperatures,
            [record[f"l{size}"]["susceptibility"] for record in records],
            marker="o",
            markersize=3,
            linestyle=style,
            label=label,
        )
        axes[2].plot(
            temperatures,
            [record[f"l{size}"]["binder_cumulant"] for record in records],
            marker="o",
            markersize=3,
            linestyle=style,
            label=label,
        )
    axes[1].axvline(
        payload["observable_route"]["chi_extrapolation"]["tc_intercept"],
        color="tab:purple",
        linestyle=":",
        label="chi 1/L extrapolation",
    )
    axes[1].set_ylabel(r"susceptibility $\chi$")
    axes[2].set_ylabel(r"Binder cumulant $U_4$")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.axvline(EXACT_TC, color="tab:red", linestyle="--", linewidth=1.3, label="exact Tc")
        axis.set_xlabel("temperature T")
        axis.grid(alpha=0.25)
    figure.suptitle("M4: two model-based routes to critical temperature")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_trajectory_figure(payload: dict[str, Any], output_path: Path) -> None:
    records = payload["observable_route"]["temperature_records"]
    temperatures = np.array([record["temperature"] for record in records])
    entropy_matrix = np.array([record["l8_policy_entropy_by_step"] for record in records])
    mean_entropy = np.mean(entropy_matrix, axis=1)
    positive_fraction = np.array([record["l8_mode_fractions"]["positive"] for record in records])
    conditional_balance = np.array(
        [record["l8_mode_fractions"]["positive_given_nonzero"] for record in records]
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    image = axes[0].imshow(
        entropy_matrix,
        origin="lower",
        aspect="auto",
        extent=[0, entropy_matrix.shape[1] - 1, temperatures[0], temperatures[-1]],
        cmap="viridis",
    )
    axes[0].axhline(EXACT_TC, color="white", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("raster assignment step")
    axes[0].set_ylabel("temperature T")
    axes[0].set_title("L=8 policy entropy by step")
    figure.colorbar(image, ax=axes[0], label="Bernoulli entropy (nats)")

    axes[1].plot(temperatures, mean_entropy, marker="o")
    axes[1].set_ylabel("mean per-step policy entropy (nats)")
    axes[1].set_xlabel("temperature T")
    axes[1].axvline(EXACT_TC, color="tab:red", linestyle="--")
    axes[1].grid(alpha=0.25)

    axes[2].plot(temperatures, positive_fraction, marker="o", label=r"P(m>0)")
    axes[2].plot(
        temperatures,
        conditional_balance,
        marker="x",
        linestyle="--",
        label=r"P(m>0 | m!=0)",
    )
    axes[2].axhline(0.5, color="black", linestyle=":", label="symmetric modes")
    axes[2].axvline(EXACT_TC, color="tab:red", linestyle="--")
    axes[2].set_xlabel("temperature T")
    axes[2].set_ylabel("positive-mode fraction")
    axes[2].set_ylim(0.4, 0.56)
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)
    figure.suptitle("M4 exploratory trajectory signatures from the L=8 policy")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"m4_metrics_{stamp}.json"
    summary_path = RESULTS_DIR / f"m4_tc_summary_{stamp}.png"
    trajectory_path = RESULTS_DIR / f"m4_trajectory_signatures_{stamp}.png"

    print("M4 validation: complete unit-test suite")
    unit_tests_passed, unit_test_metrics = run_unit_tests()
    m3_metrics_path = latest_m3_metrics()
    m3_payload = json.loads(m3_metrics_path.read_text(encoding="utf-8"))
    if m3_payload["status"] != "PASS":
        raise RuntimeError("latest M3 result is not passing")
    l4_model, l4_source = load_conditioned_model(m3_payload, "l4")
    l8_model, l8_source = load_conditioned_model(m3_payload, "l8")

    print("M4 validation: log Z differentiation route", flush=True)
    oracle = ExactIsing(4)
    log_z_temperatures = 1.0 / LOG_Z_BETA_GRID
    exact_log_z = np.array(
        [oracle.evaluate(float(temperature)).log_partition for temperature in log_z_temperatures]
    )
    with torch.no_grad():
        temperature_tensor = torch.from_numpy(log_z_temperatures.astype(np.float32))
        learned_l4_log_z = l4_model.log_z(temperature_tensor).numpy().astype(np.float64)
        learned_l8_log_z = l8_model.log_z(temperature_tensor).numpy().astype(np.float64)

    exact_curve = heat_capacity_from_log_z(
        log_z_temperatures,
        exact_log_z,
        n_sites=16,
        polynomial_degree=LOG_Z_POLYNOMIAL_DEGREE,
        dense_points=1_000,
    )
    learned_l4_curve = heat_capacity_from_log_z(
        log_z_temperatures,
        learned_l4_log_z,
        n_sites=16,
        polynomial_degree=LOG_Z_POLYNOMIAL_DEGREE,
        dense_points=1_000,
    )
    learned_l8_curve = heat_capacity_from_log_z(
        log_z_temperatures,
        learned_l8_log_z,
        n_sites=64,
        polynomial_degree=LOG_Z_POLYNOMIAL_DEGREE,
        dense_points=1_000,
    )
    direct_exact_peak = minimize_scalar(
        lambda temperature: -oracle.evaluate(float(temperature)).observables.specific_heat,
        bounds=(1.8, 2.8),
        method="bounded",
        options={"xatol": 1e-12},
    )
    direct_exact_peak_temperature = float(direct_exact_peak.x)
    exact_pipeline_error = abs(
        exact_curve.peak_temperature - direct_exact_peak_temperature
    ) / direct_exact_peak_temperature
    print(
        f"  exact direct peak={direct_exact_peak_temperature:.6f}, "
        f"pipeline peak={exact_curve.peak_temperature:.6f}, "
        f"error={exact_pipeline_error:.3%}",
        flush=True,
    )
    print(
        f"  learned peaks: L=4 T={learned_l4_curve.peak_temperature:.6f}, "
        f"L=8 T={learned_l8_curve.peak_temperature:.6f}",
        flush=True,
    )

    print("M4 validation: generated observables and L=12 Metropolis baseline", flush=True)
    temperature_records: list[dict[str, Any]] = []
    curves: dict[int, dict[str, list[float]]] = {
        size: {"susceptibility": [], "binder_cumulant": []} for size in (4, 8, 12)
    }
    for temperature_index, temperature in enumerate(TEMPERATURES):
        model_records: dict[int, dict[str, Any]] = {}
        for lattice_size, model in ((4, l4_model), (8, l8_model)):
            seed = MODEL_SAMPLE_BASE_SEEDS[lattice_size] + temperature_index
            samples, entropy_by_step = sample_with_entropy(
                model,
                float(temperature),
                MODEL_SAMPLES_PER_TEMPERATURE,
                seed,
            )
            lattices = samples.reshape(-1, lattice_size, lattice_size)
            sampled_energies = np.asarray(energy(lattices))
            sampled_magnetizations = np.asarray(magnetization(lattices))
            observables = compute_observables(
                sampled_energies,
                sampled_magnetizations,
                float(temperature),
                lattice_size**2,
            )
            positive = float(np.mean(sampled_magnetizations > 0))
            negative = float(np.mean(sampled_magnetizations < 0))
            zero = float(np.mean(sampled_magnetizations == 0))
            nonzero = positive + negative
            model_records[lattice_size] = {
                "seed": seed,
                "observables": observables.to_dict(),
                "entropy_by_step": entropy_by_step.tolist(),
                "mean_policy_entropy": float(np.mean(entropy_by_step)),
                "mode_fractions": {
                    "positive": positive,
                    "negative": negative,
                    "zero": zero,
                    "positive_given_nonzero": positive / nonzero if nonzero else 0.5,
                },
            }
            curves[lattice_size]["susceptibility"].append(observables.susceptibility)
            curves[lattice_size]["binder_cumulant"].append(observables.binder_cumulant)

        mcmc_seed = MCMC_BASE_SEED + temperature_index
        mcmc_result = run_metropolis(
            MetropolisConfig(
                lattice_size=12,
                temperature=float(temperature),
                seed=mcmc_seed,
                **MCMC_SETTINGS,
            )
        )
        curves[12]["susceptibility"].append(mcmc_result.observables.susceptibility)
        curves[12]["binder_cumulant"].append(mcmc_result.observables.binder_cumulant)
        temperature_records.append(
            {
                "temperature": float(temperature),
                "l4": model_records[4]["observables"],
                "l4_seed": model_records[4]["seed"],
                "l4_policy_entropy_by_step": model_records[4]["entropy_by_step"],
                "l4_mean_policy_entropy": model_records[4]["mean_policy_entropy"],
                "l4_mode_fractions": model_records[4]["mode_fractions"],
                "l8": model_records[8]["observables"],
                "l8_seed": model_records[8]["seed"],
                "l8_policy_entropy_by_step": model_records[8]["entropy_by_step"],
                "l8_mean_policy_entropy": model_records[8]["mean_policy_entropy"],
                "l8_mode_fractions": model_records[8]["mode_fractions"],
                "l12": mcmc_result.observables.to_dict(),
                "l12_mcmc_seed": mcmc_seed,
                "l12_mcmc_sample_count": mcmc_result.n_samples,
                "l12_mcmc_acceptance_rate": mcmc_result.acceptance_rate,
            }
        )
        print(
            f"  T={temperature:.6f}: chi L4={curves[4]['susceptibility'][-1]:.4f}, "
            f"L8={curves[8]['susceptibility'][-1]:.4f}, "
            f"L12={curves[12]['susceptibility'][-1]:.4f}",
            flush=True,
        )

    chi_peaks = {
        size: local_quadratic_peak(TEMPERATURES, curves[size]["susceptibility"])
        for size in (4, 8, 12)
    }
    inverse_sizes = 1.0 / np.array([4.0, 8.0, 12.0])
    peak_values = np.array([chi_peaks[size] for size in (4, 8, 12)])
    slope, intercept = np.polyfit(inverse_sizes, peak_values, deg=1)
    fitted_peaks = slope * inverse_sizes + intercept
    residual_sum_squares = float(np.sum((peak_values - fitted_peaks) ** 2))
    total_sum_squares = float(np.sum((peak_values - np.mean(peak_values)) ** 2))
    fit_r_squared = 1.0 - residual_sum_squares / total_sum_squares

    crossing_l4_l8 = linear_crossing(
        TEMPERATURES,
        curves[4]["binder_cumulant"],
        curves[8]["binder_cumulant"],
        (2.0, 2.6),
    )
    crossing_l8_l12 = linear_crossing(
        TEMPERATURES,
        curves[8]["binder_cumulant"],
        curves[12]["binder_cumulant"],
        (2.0, 2.6),
    )
    binder_mean = 0.5 * (crossing_l4_l8 + crossing_l8_l12)
    observable_consensus = 0.5 * (float(intercept) + binder_mean)
    print(
        f"  chi peaks {chi_peaks}; 1/L intercept={intercept:.6f}; "
        f"Binder crossings={crossing_l4_l8:.6f}, {crossing_l8_l12:.6f}; "
        f"consensus={observable_consensus:.6f}",
        flush=True,
    )

    exact_pipeline_passed = exact_pipeline_error < 0.02
    learned_log_z_tc = learned_l8_curve.peak_temperature
    learned_log_z_passed = 2.1 <= learned_log_z_tc <= 2.5
    observable_route_passed = 2.1 <= observable_consensus <= 2.5
    passed = (
        unit_tests_passed
        and exact_pipeline_passed
        and learned_log_z_passed
        and observable_route_passed
    )
    payload: dict[str, Any] = {
        "milestone": "M4",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "git": git_provenance(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "device": "cpu",
        "exact_critical_temperature": float(EXACT_TC),
        "unit_tests": {"passed": unit_tests_passed, **unit_test_metrics},
        "source_m3": {
            "metrics_path": str(m3_metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256(m3_metrics_path),
            "l4_checkpoint": l4_source,
            "l8_checkpoint": l8_source,
        },
        "log_z_route": {
            "method": {
                "beta_grid_points": int(LOG_Z_BETA_GRID.size),
                "temperature_interval": [1.5, 3.2],
                "smoothing": "degree-10 Chebyshev least-squares fit in beta",
                "identity": "C_per_site = beta^2 * d2(logZ)/d(beta)^2 / N",
                "peak_search_interval": [1.8, 2.8],
            },
            "direct_exact_l4_peak_temperature": direct_exact_peak_temperature,
            "exact_pipeline_relative_peak_error": exact_pipeline_error,
            "exact_l4_pipeline": curve_record(exact_curve),
            "learned_l4": curve_record(learned_l4_curve),
            "learned_l8": curve_record(learned_l8_curve),
            "primary_learned_tc_prediction": learned_log_z_tc,
        },
        "observable_route": {
            "model_samples_per_temperature": MODEL_SAMPLES_PER_TEMPERATURE,
            "model_sample_batch": MODEL_SAMPLE_BATCH,
            "mcmc_settings": MCMC_SETTINGS,
            "temperature_records": temperature_records,
            "chi_extrapolation": {
                "peak_method": "local quadratic through up to five points around maximum",
                "peak_temperatures": {str(size): value for size, value in chi_peaks.items()},
                "fit_model": "T_peak(L) = Tc + a/L",
                "slope": float(slope),
                "tc_intercept": float(intercept),
                "r_squared": fit_r_squared,
            },
            "binder_crossings": {
                "l4_l8": crossing_l4_l8,
                "l8_l12": crossing_l8_l12,
                "mean": binder_mean,
                "interpolation": "linear between adjacent temperature points",
            },
            "primary_consensus_tc_prediction": observable_consensus,
        },
        "criteria": {
            "a_exact_l4_log_z_pipeline_within_2_percent": exact_pipeline_passed,
            "b_learned_log_z_tc_in_2p1_to_2p5": learned_log_z_passed,
            "b_observable_tc_in_2p1_to_2p5": observable_route_passed,
            "unit_tests": unit_tests_passed,
        },
        "artifacts": {
            "summary_figure": str(summary_path.relative_to(ROOT)),
            "trajectory_figure": str(trajectory_path.relative_to(ROOT)),
        },
    }
    make_summary_figure(payload, summary_path)
    make_trajectory_figure(payload, trajectory_path)
    finished = datetime.now().astimezone()
    payload["finished_at"] = finished.isoformat()
    payload["duration_seconds"] = (finished - started).total_seconds()
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"M4 unit tests: {unit_test_metrics['tests_run']} -> {'PASS' if unit_tests_passed else 'FAIL'}")
    print(f"M4 metrics: {metrics_path.relative_to(ROOT)}")
    print(f"M4 summary: {summary_path.relative_to(ROOT)}")
    print(f"M4 trajectory diagnostics: {trajectory_path.relative_to(ROOT)}")
    print(f"M4 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

