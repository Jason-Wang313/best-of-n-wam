from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.script_contracts import audit_script_contracts, script_contracts_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical shell-script contracts.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "script_contracts.json", help="JSON output path.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "script_contracts_report.md", help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when script contract issues are found.")
    args = parser.parse_args()

    payload = audit_script_contracts(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(script_contracts_markdown(payload), encoding="utf-8")
    print(f"script contracts: verified={payload['verified']}, scripts={payload['n_scripts']}, checks={payload['n_checks']}, issues={payload['n_issues']}")
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
