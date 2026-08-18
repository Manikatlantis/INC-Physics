#!/usr/bin/env python3
"""Validate M1: exact L=4 physics and independent Metropolis baselines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inc-physics-matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.exact import ExactIsing  # noqa: E402
from src.metropolis import MetropolisConfig, run_metropolis  # noqa: E402


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
        2.60,
        2.80,
        3.00,
        3.20,
    ],
    dtype=np.float64,
)
BASE_SEED = 1_729
SAMPLER_SETTINGS = {
    4: {
        "n_chains": 192,
        "burn_sweeps": 1_000,
        "n_samples_per_chain": 2_000,
        "thin_sweeps": 2,
    },
    8: {
        "n_chains": 64,
        "burn_sweeps": 1_000,
        "n_samples_per_chain": 1_000,
        "thin_sweeps": 2,
    },
    12: {
        "n_chains": 64,
        "burn_sweeps": 1_500,
        "n_samples_per_chain": 1_500,
        "thin_sweeps": 2,
    },
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
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
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


def make_figure(
    exact_records: list[dict[str, Any]],
    mcmc_records: dict[int, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    names = [
        ("energy_per_site", r"energy per site $\langle E\rangle/N$"),
        ("abs_magnetization", r"absolute magnetization $\langle|M|\rangle/N$"),
        ("susceptibility", r"susceptibility $\chi$"),
        ("specific_heat", r"specific heat $C$"),
        ("binder_cumulant", r"Binder cumulant $U_4$"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    flat_axes = axes.ravel()
    temperatures = np.array([record["temperature"] for record in exact_records])

    for axis, (field, label) in zip(flat_axes, names, strict=False):
        exact_values = [record["observables"][field] for record in exact_records]
        axis.plot(temperatures, exact_values, color="black", linewidth=2, label="exact L=4")
        for lattice_size, records in mcmc_records.items():
            values = [record["observables"][field] for record in records]
            axis.plot(
                temperatures,
                values,
                marker="o",
                markersize=3,
                linewidth=1,
                label=f"Metropolis L={lattice_size}",
            )
        axis.axvline(EXACT_TC, color="tab:red", linestyle="--", linewidth=1.2, label="exact $T_c$")
        axis.set_xlabel("temperature T")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle("M1: exact oracle and seeded Metropolis baselines")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"m1_metrics_{stamp}.json"
    figure_path = RESULTS_DIR / f"m1_observables_{stamp}.png"

    print("M1 validation: unit tests")
    unit_tests_passed, unit_test_metrics = run_unit_tests()

    oracle = ExactIsing(4)
    exact_records: list[dict[str, Any]] = []
    for temperature in TEMPERATURES:
        exact = oracle.evaluate(float(temperature))
        exact_records.append(
            {
                "temperature": exact.temperature,
                "log_partition": exact.log_partition,
                "observables": exact.observables.to_dict(),
            }
        )

    mcmc_records: dict[int, list[dict[str, Any]]] = {4: [], 8: [], 12: []}
    print("M1 validation: Metropolis temperature sweep")
    for lattice_size in (4, 8, 12):
        settings = SAMPLER_SETTINGS[lattice_size]
        for temperature_index, temperature in enumerate(TEMPERATURES):
            seed = BASE_SEED + lattice_size * 10_000 + temperature_index
            config = MetropolisConfig(
                lattice_size=lattice_size,
                temperature=float(temperature),
                seed=seed,
                initialization="mixed",
                **settings,
            )
            result = run_metropolis(config)
            record = {
                "temperature": float(temperature),
                "seed": seed,
                "n_samples": result.n_samples,
                "acceptance_rate": result.acceptance_rate,
                "observables": result.observables.to_dict(),
            }
            mcmc_records[lattice_size].append(record)
            print(
                f"  L={lattice_size:2d} T={temperature:.6f} "
                f"E/N={result.observables.energy_per_site:.6f} "
                f"|m|={result.observables.abs_magnetization:.6f} "
                f"chi={result.observables.susceptibility:.6f} "
                f"C={result.observables.specific_heat:.6f}"
            )

    comparison_fields = {
        "energy_per_site": 0.01,
        "abs_magnetization": 0.01,
        "susceptibility": 0.05,
        "specific_heat": 0.05,
    }
    comparisons: list[dict[str, Any]] = []
    criterion_a_passed = True
    maximum_errors = {field: 0.0 for field in comparison_fields}
    for exact_record, mcmc_record in zip(exact_records, mcmc_records[4], strict=True):
        errors: dict[str, float] = {}
        within_tolerance: dict[str, bool] = {}
        for field, tolerance in comparison_fields.items():
            error = relative_error(
                mcmc_record["observables"][field],
                exact_record["observables"][field],
            )
            errors[field] = error
            maximum_errors[field] = max(maximum_errors[field], error)
            within_tolerance[field] = error <= tolerance
        temperature_passed = all(within_tolerance.values())
        criterion_a_passed &= temperature_passed
        comparisons.append(
            {
                "temperature": exact_record["temperature"],
                "relative_errors": errors,
                "within_tolerance": within_tolerance,
                "passed": temperature_passed,
            }
        )

    l12_susceptibilities = np.array(
        [record["observables"]["susceptibility"] for record in mcmc_records[12]]
    )
    peak_index = int(np.argmax(l12_susceptibilities))
    l12_peak_temperature = float(TEMPERATURES[peak_index])
    criterion_b_passed = 2.15 <= l12_peak_temperature <= 2.45
    passed = unit_tests_passed and criterion_a_passed and criterion_b_passed

    make_figure(exact_records, mcmc_records, figure_path)
    finished = datetime.now().astimezone()
    payload = {
        "milestone": "M1",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "seed": BASE_SEED,
        "git": git_provenance(),
        "python": sys.version,
        "exact_critical_temperature": float(EXACT_TC),
        "temperature_grid": TEMPERATURES.tolist(),
        "sampler_settings": SAMPLER_SETTINGS,
        "unit_tests": {"passed": unit_tests_passed, **unit_test_metrics},
        "criteria": {
            "a_l4_mcmc_vs_exact": {
                "passed": criterion_a_passed,
                "tolerances": comparison_fields,
                "maximum_relative_errors": maximum_errors,
                "by_temperature": comparisons,
            },
            "b_l12_susceptibility_peak": {
                "passed": criterion_b_passed,
                "peak_temperature": l12_peak_temperature,
                "accepted_interval": [2.15, 2.45],
            },
            "c_unit_tests": {"passed": unit_tests_passed},
        },
        "exact_l4": exact_records,
        "metropolis": {str(size): records for size, records in mcmc_records.items()},
        "artifacts": {"figure": str(figure_path.relative_to(ROOT))},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nM1 acceptance criteria")
    print(
        "  (a) L=4 maximum relative errors: "
        + ", ".join(f"{name}={value:.3%}" for name, value in maximum_errors.items())
        + f" -> {'PASS' if criterion_a_passed else 'FAIL'}"
    )
    print(
        f"  (b) L=12 chi peak T={l12_peak_temperature:.6f} "
        f"in [2.15, 2.45] -> {'PASS' if criterion_b_passed else 'FAIL'}"
    )
    print(f"  (c) {unit_test_metrics['tests_run']} unit tests -> {'PASS' if unit_tests_passed else 'FAIL'}")
    print(f"  metrics: {json_path.relative_to(ROOT)}")
    print(f"  figure:  {figure_path.relative_to(ROOT)}")
    print(f"M1 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
