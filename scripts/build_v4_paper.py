from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "iclr_submission.tex"
PDF = ROOT / "iclr_submission.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "best-of-n-wam-v4.pdf"
FINAL_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME
OLD_FINALS = [
    ROOT / "paper" / "final" / "best-of-n-wam-v3.pdf",
    DESKTOP / "best-of-n-wam-v3.pdf",
]


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    env.setdefault("FORCE_SOURCE_DATE", "1")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    run(["python", "experiments/v4_frozen_evidence.py"])
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
    shutil.copy2(PDF, DESKTOP_PDF)
    for old_pdf in OLD_FINALS:
        if old_pdf.exists():
            old_pdf.unlink()
    print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()
