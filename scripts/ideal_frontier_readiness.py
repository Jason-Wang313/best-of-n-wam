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

from wam_inference_value.ideal_frontier_readiness import (  # noqa: E402
    audit_ideal_frontier_readiness,
    ideal_frontier_readiness_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit readiness of future-only ideal robotics frontiers.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output = args.output or results_dir / "ideal_frontier_readiness.json"
    report = args.report or ROOT / "reports" / "ideal_frontier_readiness_report.md"

    payload = audit_ideal_frontier_readiness(ROOT, results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(ideal_frontier_readiness_markdown(payload), encoding="utf-8")
    print(
        "ideal frontier readiness: "
        f"verified={payload['verified']}, frontiers={payload['n_frontiers']}, "
        f"ready={payload['n_ready_to_promote']}, not_ready={payload['n_not_ready_to_promote']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
