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

V5_CACHE_FILES = [
    RESULTS / "v5" / "summary.json",
    RESULTS / "v5" / "prospective_evidence_summary.json",
    RESULTS / "v5_frozen_evidence" / "summary.json",
    ROOT / "v5_results_macros.tex",
    ROOT / "paper" / "v5_summary_table.tex",
    ROOT / "paper" / "v5_compute_frontier_table.tex",
    ROOT / "scoretailbench" / "manifest.json",
]

V5_FIGURES = [
    ROOT / "paper_figures" / "v5" / "v5_census_regimes.pdf",
    ROOT / "paper_figures" / "v5" / "v5_closed_loop_returns.pdf",
    ROOT / "paper_figures" / "v5" / "v5_equal_compute_frontier.pdf",
    ROOT / "paper_figures" / "v5" / "v5_label_budget.pdf",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_v5_artifacts() -> dict[str, Any]:
    for path in V5_CACHE_FILES:
        require(path.exists(), f"missing V5 cache file: {path}")
    for path in V5_FIGURES:
        require(path.exists() and path.stat().st_size > 1000, f"missing or empty V5 figure: {path}")

    core = load_json(RESULTS / "v5" / "summary.json")
    prospective = load_json(RESULTS / "v5" / "prospective_evidence_summary.json")
    frozen = load_json(RESULTS / "v5_frozen_evidence" / "summary.json")
    require(core.get("gate_passed") is True, "V5 core gate failed")
    require(prospective.get("gate_passed") is True, "V5 prospective gate failed")
    require(frozen.get("gate_passed") is True, "V5 frozen gate failed")
    require((core.get("finite_pool_census") or {}).get("row_count") == 1458, "V5 census row count changed")
    require((prospective.get("prospective_audit") or {}).get("heldout_pools") == 50, "V5 heldout pool count changed")
    require((prospective.get("closed_loop_validation") or {}).get("gate_passed") is True, "V5 closed-loop gate failed")
    return frozen


def check_claim_status() -> dict[str, Any]:
    completed = run([sys.executable, "scripts/claims_status.py"], check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        fail("claims_status.py reported overclaims or failed")
    payload = load_json(RESULTS / "claims_status.json")
    require(payload.get("num_verified", 0) >= 138, "V5 claim ledger verified count regressed")
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
    frozen = check_v5_artifacts()
    claims = check_claim_status()
    build_pdf()
    check_latex_log()
    print("WAM v5 audit passed")
    print(
        f"verified_claims={claims.get('num_verified')} "
        f"prospective_pools={frozen.get('prospective_pools')} "
        f"pdf_bytes={PDF.stat().st_size}"
    )


if __name__ == "__main__":
    main()
