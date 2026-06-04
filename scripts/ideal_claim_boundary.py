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

from wam_inference_value.ideal_claim_boundary import (  # noqa: E402
    audit_ideal_claim_boundary,
    ideal_claim_boundary_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ideal claims as promotable results or future-only non-claims.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when the boundary audit fails.")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output = args.output or results_dir / "ideal_claim_boundary.json"
    report = args.report or ROOT / "reports" / "ideal_claim_boundary_report.md"

    payload = audit_ideal_claim_boundary(ROOT, results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(ideal_claim_boundary_markdown(payload), encoding="utf-8")
    print(
        "ideal claim boundary: "
        f"verified={payload['verified']}, ideal={payload['n_ideal_claims']}, "
        f"promotable={payload['n_promotable_claims']}, future_only={payload['n_future_only_claims']}, "
        f"all_promotable={payload['all_ideal_claims_promotable']}, issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
