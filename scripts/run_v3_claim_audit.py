from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "best-of-n-wam-v3.pdf"
REPO_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SUMMARY = ROOT / "results" / "v3_cached_evidence" / "summary.json"
CLAIMS_STATUS = ROOT / "results" / "claims_status.json"
LATEX_LOG = ROOT / "iclr_submission.log"


EXPECTED_CACHE_FILES = [
    ROOT / "results" / "v3_cached_evidence" / "summary.json",
    ROOT / "results" / "v3_cached_evidence" / "v3_core_claims.csv",
    ROOT / "results" / "v3_cached_evidence" / "v3_failure_modes.csv",
    ROOT / "results" / "v3_cached_evidence" / "v3_benchmark_summary.csv",
    ROOT / "results" / "v3_cached_evidence" / "v3_coverage_summary.csv",
    ROOT / "v3_results_macros.tex",
]

EXPECTED_FIGURES = [
    "v3_exact_law_errors.pdf",
    "v3_benchmark_ci_lowers.pdf",
    "v3_failure_modes.pdf",
    "v3_robocasa_coverage.pdf",
    "v3_claim_artifact_counts.pdf",
]

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    if not path.exists():
        fail(f"missing PDF: {path}")
    output = run(["pdfinfo", str(path)]).stdout
    match = re.search(r"^Pages:\s+(\d+)", output, re.MULTILINE)
    if not match:
        fail(f"could not read page count for {path}")
    return int(match.group(1))


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_summary(summary: dict[str, Any]) -> None:
    require(summary.get("claims_verified", 0) >= 127, "too few verified claims")
    require(summary.get("claims_partial") == 0, "partial claims present")
    require(summary.get("claims_unsupported") == 0, "unsupported claims present")
    require(summary.get("artifact_files", 0) >= 462, "artifact file count regressed")
    require(summary.get("artifact_csv", 0) >= 232, "CSV artifact count regressed")
    require(summary.get("artifact_json", 0) >= 149, "JSON artifact count regressed")
    require(summary.get("success_mae", 1.0) < 0.003, "success MAE too high")
    require(summary.get("utility_mae", 1.0) < 0.02, "utility MAE too high")
    require(abs(summary.get("auc_identity_error", 1.0)) <= 1e-12, "AUC identity error regressed")
    require(summary.get("same_p_kappa_gap_n64", 0.0) > 0.9, "same-AUC high-N gap too small")
    require(summary.get("severe_gap_growth", 0.0) > 10.0, "severe mismatch gap too small")
    require(summary.get("stuck_gap_growth", 0.0) > 10.0, "stuck/slip mismatch gap too small")
    require(summary.get("anti_scorer_n64", 0.0) < summary.get("anti_scorer_n1", 0.0), "anti-scorer did not worsen")
    require(summary.get("adaptive_gain", 0.0) > 0.05, "adaptive allocation gain too small")
    require(summary.get("fetch_learned_ci_lo", 0.0) > 0.3, "Fetch learned CI lower too small")
    require(summary.get("fetch_rgb_ci_lo", 0.0) > 0.2, "Fetch RGB CI lower too small")
    require(summary.get("robocasa97_tasks", 0) >= 97, "RoboCasa 97-task coverage regressed")
    require(summary.get("robocasa97_pools", 0) >= 194, "RoboCasa pool count regressed")
    require(summary.get("robocasa97_ci_lo", 0.0) > 0.2, "RoboCasa 97-task CI lower too small")
    require(summary.get("robocasa35_tasks", 0) >= 35, "RoboCasa residual coverage regressed")
    require(summary.get("robocasa35_ci_lo", 0.0) > 0.1, "RoboCasa residual CI lower too small")
    require(summary.get("libero_ci_lo", 0.0) > 0.1, "LIBERO CI lower too small")
    require(summary.get("benchmark_rows") == 8, "benchmark row count changed")


def check_claim_status() -> None:
    completed = run([sys.executable, "scripts/claims_status.py"], check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        fail("claims_status.py reported overclaims or failed")
    payload = load_json(CLAIMS_STATUS)
    require(payload.get("num_verified", 0) >= 127, "claim ledger verified count regressed")
    require(payload.get("num_partial") == 0, "claim ledger has partial claims")
    require(payload.get("num_unsupported") == 0, "claim ledger has unsupported claims")
    require(payload.get("num_failed") == 0, "claim ledger has failed claims")
    for key in ["readme_overclaims", "paper_overclaims", "report_overclaims", "narrative_overclaims", "overclaims"]:
        require(not payload.get(key), f"claim ledger reports {key}")


def check_cache_files() -> None:
    for path in EXPECTED_CACHE_FILES:
        require(path.exists(), f"missing v3 cache file: {path}")
    for name in EXPECTED_FIGURES:
        require((ROOT / "results" / "v3_cached_evidence" / "figures" / name).exists(), f"missing evidence figure {name}")
        require((ROOT / "paper_figures" / "v3" / name).exists(), f"missing paper figure {name}")


def check_source_map() -> None:
    require(SOURCE_MAP.exists(), f"missing source map: {SOURCE_MAP}")
    text = SOURCE_MAP.read_text(encoding="utf-8")
    expected = f"| `{FINAL_NAME}` | `{ROOT}` | `Jason-Wang313/best-of-n-wam` |"
    require(expected in text, "source map does not point WAM to v3 final")
    require("best-of-n-wam-v2.pdf" not in text, "source map still contains WAM v2")


def check_latex_log() -> None:
    require(LATEX_LOG.exists(), "missing LaTeX log")
    text = LATEX_LOG.read_text(encoding="utf-8", errors="replace")
    blockers = [pattern for pattern in LOG_BLOCKERS if re.search(pattern, text, re.IGNORECASE)]
    require(not blockers, f"LaTeX log blockers present: {blockers}")


def check_git_tracking() -> None:
    tracked_build_pdf = run(["git", "ls-files", "--error-unmatch", "iclr_submission.pdf"], check=False)
    require(tracked_build_pdf.returncode != 0, "generated iclr_submission.pdf is still tracked")


def main() -> None:
    run([sys.executable, "experiments/v3_cached_evidence.py"])
    check_cache_files()
    summary = load_json(SUMMARY)
    check_summary(summary)
    check_claim_status()

    repo_pages = pdf_pages(REPO_PDF)
    desktop_pages = pdf_pages(DESKTOP_PDF)
    require(repo_pages >= 25, f"repo final PDF has only {repo_pages} pages")
    require(desktop_pages >= 25, f"Desktop final PDF has only {desktop_pages} pages")
    require(sha256(REPO_PDF) == sha256(DESKTOP_PDF), "repo and Desktop PDFs differ")
    require(not (DESKTOP / "best-of-n-wam-v2.pdf").exists(), "stale Desktop v2 PDF exists")

    check_source_map()
    check_latex_log()
    check_git_tracking()

    print("WAM v3 audit passed")
    print(f"pages={repo_pages} sha256={sha256(REPO_PDF)}")


if __name__ == "__main__":
    main()
