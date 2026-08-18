#!/usr/bin/env python3
"""Validate M5 report completeness, traceability, and citation allowlist."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "REPORT.md"
RESULTS_DIR = ROOT / "results"
REQUIRED_HEADINGS = (
    "## Abstract",
    "## 1. Problem statement",
    "## 2. Ising and Boltzmann background",
    "## 3. GFlowNet formulation",
    "## 4. Validation methodology",
    "## 5. Results",
    "## 6. Discussion and limitations",
    "## 7. Follow-up work",
    "## 8. Reproducibility map",
    "## 9. Conclusion",
    "## References",
)
REQUIRED_ARTIFACTS = (
    "results/m1_metrics_20260818T002648-0400.json",
    "results/m2_metrics_20260818T012603-0400.json",
    "results/m3_metrics_20260818T015650-0400.json",
    "results/m4_metrics_20260818T021212-0400.json",
    "results/m1_observables_20260818T002648-0400.png",
    "results/m2_fixed_temperature_20260818T012603-0400.png",
    "results/m3_conditioned_summary_20260818T015650-0400.png",
    "results/m4_tc_summary_20260818T021212-0400.png",
    "results/m4_trajectory_signatures_20260818T021212-0400.png",
)
REQUIRED_RESULT_STRINGS = (
    "0.014762",
    "0.015377",
    "0.003255",
    "0.498825",
    "2.34 +/- 0.15",
    "6.5%",
    "2.273526",
    "2.246221",
    "2.259472",
    "2.25 to 2.27",
    "multinomial draws from exactly enumerated model probabilities",
    "true sequential rollouts",
    "opposite signs",
    "overstates precision",
    "one automated session of roughly 90 minutes",
    "0.0126%",
)
ALLOWED_ARXIV_IDS = {"2111.09266", "2201.13259", "2202.01361"}


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"m5_metrics_{stamp}.json"

    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    heading_checks = {heading: heading in report for heading in REQUIRED_HEADINGS}
    artifact_checks = {
        path: path in report and (ROOT / path).is_file() for path in REQUIRED_ARTIFACTS
    }
    result_string_checks = {value: value in report for value in REQUIRED_RESULT_STRINGS}
    linked_local_paths = re.findall(r"\]\((?!https?://|#)([^)]+)\)", report)
    broken_links = sorted(
        {
            path
            for path in linked_local_paths
            if not (ROOT / path).exists()
        }
    )
    cited_arxiv_ids = set(re.findall(r"arXiv:(\d{4}\.\d{5})", report))
    disallowed_arxiv_ids = sorted(cited_arxiv_ids - ALLOWED_ARXIV_IDS)
    placeholder_matches = re.findall(r"\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b", report, re.I)
    word_count = len(re.findall(r"\b\w+[\w'-]*\b", report))

    limitation_terms = {
        "cpu_bound": "CPU-bound capacity" in report,
        "finite_size": "Finite sizes" in report,
        "mode_coverage": "Mode coverage" in report,
        "mcmc_advantage": "MCMC still wins" in report,
        "negative_heat_capacity_artifact": bool(
            re.search(r"nonphysical\s+negative\s+differentiated\s+heat\s+capacity", report)
        ),
    }
    checks = {
        "report_exists": REPORT_PATH.is_file(),
        "all_required_headings": all(heading_checks.values()),
        "all_required_artifacts_linked_and_exist": all(artifact_checks.values()),
        "all_key_result_strings_present": all(result_string_checks.values()),
        "no_broken_local_links": not broken_links,
        "citations_within_allowlist": not disallowed_arxiv_ids and "Onsager" in report,
        "no_placeholders": not placeholder_matches,
        "all_required_limitations": all(limitation_terms.values()),
        "substantive_length": word_count >= 2_000,
    }
    passed = all(checks.values())
    finished = datetime.now().astimezone()
    payload = {
        "milestone": "M5",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "git": git_provenance(),
        "python": sys.version,
        "report": {
            "path": str(REPORT_PATH.relative_to(ROOT)),
            "sha256": sha256(REPORT_PATH) if REPORT_PATH.exists() else None,
            "word_count": word_count,
        },
        "checks": checks,
        "details": {
            "headings": heading_checks,
            "artifacts": artifact_checks,
            "key_result_strings": result_string_checks,
            "broken_local_links": broken_links,
            "cited_arxiv_ids": sorted(cited_arxiv_ids),
            "disallowed_arxiv_ids": disallowed_arxiv_ids,
            "placeholder_matches": placeholder_matches,
            "limitations": limitation_terms,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"M5 report words: {word_count}")
    for name, check_passed in checks.items():
        print(f"  {name}: {'PASS' if check_passed else 'FAIL'}")
    print(f"M5 metrics: {output_path.relative_to(ROOT)}")
    print(f"M5 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
