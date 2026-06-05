from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import results_dir
from wam_inference_value.ideal_frontier_blocker_audit import (
    build_ideal_frontier_blocker_audit,
    ideal_frontier_blocker_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "ideal_frontier_blocker_audit_report.md")
    args = parser.parse_args()

    payload = build_ideal_frontier_blocker_audit(ROOT, args.results_dir or results_dir())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(ideal_frontier_blocker_markdown(payload), encoding="utf-8")
    print(
        "ideal frontier blocker audit: "
        f"verified={payload.get('verified')}, "
        f"frontiers={payload.get('n_frontiers')}, "
        f"ready={payload.get('n_ready_to_promote')}, "
        f"issues={payload.get('n_issues')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
