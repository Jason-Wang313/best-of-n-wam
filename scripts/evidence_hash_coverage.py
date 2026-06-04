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

from wam_inference_value.evidence_hash_coverage import (  # noqa: E402
    audit_evidence_hash_coverage,
    evidence_hash_coverage_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash current claim evidence sources and published artifact references.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when evidence hash issues are found.")
    args = parser.parse_args()

    results_override = os.environ.get("WAM_RESULTS_DIR")
    results_dir = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    output = args.output or results_dir / "evidence_hash_coverage.json"
    report = args.report or ROOT / "reports" / "evidence_hash_coverage_report.md"

    payload = audit_evidence_hash_coverage(ROOT, results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(evidence_hash_coverage_markdown(payload), encoding="utf-8")
    print(
        "evidence hash coverage: "
        f"verified={payload['verified']}, claim_sources={payload['n_claim_sources']}, "
        f"artifact_refs={payload['n_artifact_references']}, hashed={payload['n_hashed_records']}, "
        f"issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
