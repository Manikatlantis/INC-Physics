#!/usr/bin/env python3
"""Build and validate the M6 academic paper PDF."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
BUILDER = ROOT / "paper/build_paper.py"
REQUIRED_TEXT = (
    "Abstract",
    "1. Introduction",
    "2. Model and Methods",
    "3. Validation Results",
    "4. Predicting Critical Temperature",
    "5. Discussion",
    "6. Conclusion",
    "References",
    "2.3422",
    "2.2599",
    "2.2692",
    "nonphysical boundary curvature",
    "MCMC still wins",
    "Artifact Provenance",
)
REQUIRED_ARTIFACT_TEXT = (
    "results/m1_observables_20260818T002648-0400.png",
    "results/m2_fixed_temperature_20260818T012603-0400.png",
    "results/m3_conditioned_summary_20260818T015650-0400.png",
    "results/m4_tc_summary_20260818T021212-0400.png",
    "results/m4_trajectory_signatures_20260818T021212-0400.png",
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


def count_pdf_images(reader: PdfReader) -> int:
    count = 0
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        resources = resources.get_object()
        xobjects = resources.get("/XObject")
        if xobjects is None:
            continue
        for reference in xobjects.get_object().values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Image":
                count += 1
    return count


def main() -> int:
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%dT%H%M%S%z")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = RESULTS_DIR / f"m6_ising_gflownet_paper_{stamp}.pdf"
    metrics_path = RESULTS_DIR / f"m6_metrics_{stamp}.json"

    build_process = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(pdf_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    build_clean = build_process.returncode == 0 and pdf_path.is_file()
    page_count = 0
    image_count = 0
    extracted_text = ""
    metadata: dict[str, str | None] = {}
    if build_clean:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        image_count = count_pdf_images(reader)
        metadata = {
            "title": reader.metadata.title if reader.metadata else None,
            "author": reader.metadata.author if reader.metadata else None,
            "subject": reader.metadata.subject if reader.metadata else None,
        }

    normalized_text = " ".join(extracted_text.split())
    text_checks = {phrase: phrase in normalized_text for phrase in REQUIRED_TEXT}
    artifact_checks = {
        artifact: artifact in normalized_text and (ROOT / artifact).is_file()
        for artifact in REQUIRED_ARTIFACT_TEXT
    }
    cited_arxiv_ids = set(re.findall(r"arXiv:(\d{4}\.\d{5})", extracted_text))
    disallowed_arxiv_ids = sorted(cited_arxiv_ids - ALLOWED_ARXIV_IDS)
    placeholder_matches = re.findall(
        r"\b(?:TODO|TBD|FIXME|PLACEHOLDER|undefined reference)\b",
        extracted_text,
        flags=re.I,
    )
    word_count = len(re.findall(r"\b\w+[\w'-]*\b", extracted_text))
    checks = {
        "build_exit_zero": build_clean,
        "academic_page_count": 6 <= page_count <= 16,
        "all_required_sections_and_claims": all(text_checks.values()),
        "all_five_figures_provenanced": all(artifact_checks.values()),
        "all_five_figures_embedded": image_count >= 5,
        "citation_allowlist": not disallowed_arxiv_ids and cited_arxiv_ids == ALLOWED_ARXIV_IDS,
        "no_placeholders_or_orphan_markers": not placeholder_matches,
        "substantive_extracted_text": word_count >= 1_500,
        "metadata_present": bool(metadata.get("title") and metadata.get("author") and metadata.get("subject")),
    }
    passed = all(checks.values())
    finished = datetime.now().astimezone()
    payload = {
        "milestone": "M6",
        "status": "PASS" if passed else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "git": git_provenance(),
        "python": sys.version,
        "builder": {
            "path": str(BUILDER.relative_to(ROOT)),
            "exit_code": build_process.returncode,
            "stdout": build_process.stdout.strip(),
            "stderr": build_process.stderr.strip(),
            "engine": "ReportLab two-column template",
        },
        "pdf": {
            "path": str(pdf_path.relative_to(ROOT)),
            "sha256": sha256(pdf_path) if pdf_path.is_file() else None,
            "bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
            "pages": page_count,
            "embedded_images": image_count,
            "extracted_word_count": word_count,
            "metadata": metadata,
        },
        "checks": checks,
        "details": {
            "required_text": text_checks,
            "artifact_text": artifact_checks,
            "cited_arxiv_ids": sorted(cited_arxiv_ids),
            "disallowed_arxiv_ids": disallowed_arxiv_ids,
            "placeholder_matches": placeholder_matches,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"M6 PDF: {pdf_path.relative_to(ROOT)}")
    print(f"M6 pages={page_count}, embedded_images={image_count}, extracted_words={word_count}")
    for name, check_passed in checks.items():
        print(f"  {name}: {'PASS' if check_passed else 'FAIL'}")
    print(f"M6 metrics: {metrics_path.relative_to(ROOT)}")
    print(f"M6 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
