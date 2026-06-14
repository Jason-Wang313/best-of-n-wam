from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "iclr_submission.tex"
PDF = ROOT / "iclr_submission.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "best-of-n-wam-v3.pdf"
FINAL_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run(["python", "experiments/v3_cached_evidence.py"])
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
    print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()
