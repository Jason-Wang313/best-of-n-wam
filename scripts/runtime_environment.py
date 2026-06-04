from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.runtime_environment import build_and_audit_runtime_environment, runtime_environment_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Write and verify the runtime dependency environment.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when runtime checks fail.")
    args = parser.parse_args()

    output = args.output or ROOT / "results" / "runtime_environment.json"
    report = args.report or ROOT / "reports" / "runtime_environment_report.md"

    payload = build_and_audit_runtime_environment(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(runtime_environment_markdown(payload), encoding="utf-8")
    print(
        "runtime environment: "
        f"verified={payload['verified']}, python={payload['python']['version']}, "
        f"core={payload['n_core_requirements']}, optional_available={payload['n_optional_available']}, "
        f"checks={payload['n_checks']}, issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
