from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.train_inference_optimizer import (  # noqa: E402
    optimize_train_inference,
    optimizer_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-bound train/inference optimizer over committed WAM artifacts."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    reports_override = os.environ.get("WAM_REPORTS_DIR")
    reports_dir = Path(reports_override).expanduser() if reports_override else ROOT / "reports"
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir

    output = args.output or results_dir / "universal_wam_train_inference_optimizer.json"
    report = args.report or reports_dir / "universal_wam_train_inference_optimizer_report.md"
    payload = optimize_train_inference(ROOT, results_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(optimizer_markdown(payload), encoding="utf-8")

    print(
        "train/inference optimizer: "
        f"verified={payload['verified']}, candidates={payload['n_candidates']}, "
        f"envs={payload['n_selected_environments']}, families={payload['n_environment_families']}, "
        f"issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
