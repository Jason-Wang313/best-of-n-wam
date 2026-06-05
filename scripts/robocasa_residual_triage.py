from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import results_dir
from wam_inference_value.robocasa_residual_triage import (
    build_robocasa_residual_triage,
    robocasa_residual_triage_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "robocasa_residual_triage_report.md")
    args = parser.parse_args()
    payload = build_robocasa_residual_triage(ROOT, args.results_dir or results_dir())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(robocasa_residual_triage_markdown(payload), encoding="utf-8")
    print(
        "robocasa residual triage: "
        f"verified={payload.get('verified')}, "
        f"registry={payload.get('registry_count')}, "
        f"any_covered={payload.get('any_covered')}, "
        f"attempted_not_covered={payload.get('attempted_not_covered')}, "
        f"unattempted={payload.get('unattempted')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
