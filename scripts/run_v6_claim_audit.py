from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PDF = ROOT / "iclr_submission.pdf"
LATEX_LOG = ROOT / "iclr_submission.log"

LOG_BLOCKERS = [
    r"! LaTeX Error",
    r"Undefined control sequence",
    r"Emergency stop",
    r"Fatal error",
    r"Counter too large",
    r"Overfull",
    r"Citation .*undefined",
    r"Reference .*undefined",
    r"There were undefined",
    r"Label\(s\) may have changed",
    r"Rerun to get",
]

V6_CACHE_FILES = [
    RESULTS / "v6" / "summary.json",
    RESULTS / "v6" / "real_benchmark_audit_predictions.csv",
    RESULTS / "v6" / "real_benchmark_audit_predictions.sha256",
    RESULTS / "v6" / "real_benchmark_audit_outcomes.csv",
    RESULTS / "v6" / "cross_family_transfer.csv",
    RESULTS / "v6" / "selector_metric_ablation.csv",
    RESULTS / "v6" / "robustness_grid.csv",
    RESULTS / "v6" / "finite_sample_audit_theory.csv",
    RESULTS / "v6_frozen_evidence" / "summary.json",
    ROOT / "v6_results_macros.tex",
    ROOT / "paper" / "v6_summary_table.tex",
    ROOT / "paper" / "v6_ablation_table.tex",
]

V6_FIGURES = [
    ROOT / "paper_figures" / "v6" / "v6_cross_family_accuracy.pdf",
    ROOT / "paper_figures" / "v6" / "v6_cross_family_decided_rate.pdf",
    ROOT / "paper_figures" / "v6" / "v6_compute_ablation.pdf",
    ROOT / "paper_figures" / "v6" / "v6_robustness_grid.pdf",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_v6_artifacts() -> dict[str, Any]:
    for path in V6_CACHE_FILES:
        require(path.exists(), f"missing V6 cache file: {path}")
    for path in V6_FIGURES:
        require(path.exists() and path.stat().st_size > 1000, f"missing or empty V6 figure: {path}")
    summary = load_json(RESULTS / "v6" / "summary.json")
    frozen = load_json(RESULTS / "v6_frozen_evidence" / "summary.json")
    audit = summary["real_benchmark_audit"]
    require(summary.get("gate_passed") is True, "V6 evidence gate failed")
    require(frozen.get("gate_passed") is True, "V6 frozen gate failed")
    require(audit.get("family_count", 0) >= 10, "V6 family count regressed")
    require(audit.get("pool_count", 0) >= 500, "V6 pool count regressed")
    require((audit.get("decision_accuracy") or 0.0) >= 0.95, "V6 decision accuracy regressed")
    require((audit.get("false_allow_rate") or 1.0) <= 0.01, "V6 false allow regressed")
    return summary


def check_claim_status() -> dict[str, Any]:
    completed = run([sys.executable, "scripts/claims_status.py"], check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        fail("claims_status.py reported overclaims or failed")
    payload = load_json(RESULTS / "claims_status.json")
    require(payload.get("num_verified", 0) >= 150, "V6 claim ledger verified count regressed")
    require(payload.get("num_partial") == 0, "claim ledger has partial claims")
    require(payload.get("num_unsupported") == 0, "claim ledger has unsupported claims")
    require(payload.get("num_failed") == 0, "claim ledger has failed claims")
    require(not payload.get("overclaims"), "claim ledger reports overclaims")
    return payload


def build_pdf() -> None:
    for suffix in [".aux", ".bbl", ".blg", ".log", ".out", ".pdf"]:
        target = ROOT / f"iclr_submission{suffix}"
        if target.exists():
            target.unlink()
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "iclr_submission.tex"])
    run(["bibtex", "iclr_submission"])
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "iclr_submission.tex"])
    require(PDF.exists() and PDF.stat().st_size > 100_000, f"missing built PDF: {PDF}")


def check_latex_log() -> None:
    require(LATEX_LOG.exists(), "missing LaTeX log")
    text = LATEX_LOG.read_text(encoding="utf-8", errors="replace")
    blockers = [pattern for pattern in LOG_BLOCKERS if re.search(pattern, text, re.IGNORECASE)]
    require(not blockers, f"LaTeX log blockers present: {blockers}")


def main() -> None:
    run([sys.executable, "experiments/v4_frozen_evidence.py"])
    run([sys.executable, "experiments/v5_frozen_evidence.py"])
    run([sys.executable, "experiments/v6_frozen_evidence.py"])
    summary = check_v6_artifacts()
    claims = check_claim_status()
    build_pdf()
    check_latex_log()
    print("WAM v6 audit passed")
    print(
        f"verified_claims={claims.get('num_verified')} "
        f"families={summary['real_benchmark_audit']['family_count']} "
        f"pools={summary['real_benchmark_audit']['pool_count']} "
        f"pdf_bytes={PDF.stat().st_size}"
    )


if __name__ == "__main__":
    main()
