from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.artifact_integrity import audit_result_artifacts, artifact_integrity_markdown


def main() -> None:
    results_override = os.environ.get("WAM_RESULTS_DIR")
    default_results = Path(results_override).expanduser() if results_override else ROOT / "results"
    if not default_results.is_absolute():
        default_results = ROOT / default_results
    parser = argparse.ArgumentParser(description="Audit result JSON artifact references.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--output-json", type=Path, default=default_results / "artifact_integrity.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "artifact_integrity_report.md")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    payload = audit_result_artifacts(args.root, args.results_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(artifact_integrity_markdown(payload), encoding="utf-8")
    print(
        "artifact integrity: "
        f"verified={payload['verified']}, "
        f"references={payload['n_references']}, "
        f"issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
