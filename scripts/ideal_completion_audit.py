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

from wam_inference_value.ideal_completion_audit import (  # noqa: E402
    audit_ideal_completion,
    ideal_completion_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether every ideal endpoint is currently artifact-supported.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if the audit is inconsistent.")
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero unless all ideal endpoints are supported.")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output = args.output or results_dir / "ideal_completion_audit.json"
    report = args.report or ROOT / "reports" / "ideal_completion_audit_report.md"

    payload = audit_ideal_completion(ROOT, results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(ideal_completion_markdown(payload), encoding="utf-8")
    print(
        "ideal completion audit: "
        f"verified={payload['verified']}, verdict={payload['completion_verdict']}, "
        f"supported={payload['n_supported_endpoints']}, unsupported={payload['n_unsupported_endpoints']}, "
        f"future_blockers={payload['n_future_blockers']}, issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)
    if args.require_complete and not payload["all_ideal_endpoints_supported"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
