from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "iclr_submission.tex"
PDF = ROOT / "iclr_submission.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "best-of-n-wam-v6.pdf"
FINAL_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    env.setdefault("FORCE_SOURCE_DATE", "1")
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        print(completed.stdout)
        completed.check_returncode()


def main() -> None:
    run(["python", "experiments/v4_frozen_evidence.py"])
    run(["python", "experiments/v5_frozen_evidence.py"])
    run(["python", "experiments/v6_frozen_evidence.py"])
    run(["python", "experiments/libero_main_paper_summary.py"])
    run(["python", "scripts/claims_status.py"])
    for suffix in [".aux", ".bbl", ".blg", ".log", ".out", ".pdf"]:
        target = ROOT / f"iclr_submission{suffix}"
        if target.exists():
            target.unlink()

    run(["pdflatex", "-interaction=nonstopmode", TEX.name])
    run(["bibtex", "iclr_submission"])
    for _ in range(4):
        run(["pdflatex", "-interaction=nonstopmode", TEX.name])

    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, FINAL_PDF)
    if DESKTOP.exists():
        shutil.copy2(PDF, DESKTOP_PDF)
        print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()
