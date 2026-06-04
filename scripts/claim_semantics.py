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

from wam_inference_value.claim_semantics import audit_claim_semantics, claim_semantics_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit verified claim wording against semantic evidence thresholds.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when claim semantic issues are found.")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output = args.output or results_dir / "claim_semantics.json"
    report = args.report or ROOT / "reports" / "claim_semantics_report.md"

    payload = audit_claim_semantics(ROOT, results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(claim_semantics_markdown(payload), encoding="utf-8")
    print(
        "claim semantics: "
        f"verified={payload['verified']}, claims={payload['n_claims']}, checks={payload['n_checks']}, "
        f"issues={payload['n_issues']}, ci_claims={payload['n_ci_claims']}, positive_ci_claims={payload['n_positive_ci_claims']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
