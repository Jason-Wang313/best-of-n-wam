from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.model_artifact_integrity import audit_model_artifacts, model_artifact_integrity_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify committed learned-model artifacts are loadable and sane.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Model directory to audit.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when model-artifact checks fail.")
    args = parser.parse_args()

    output = args.output or ROOT / "results" / "model_artifact_integrity.json"
    report = args.report or ROOT / "reports" / "model_artifact_integrity_report.md"
    payload = audit_model_artifacts(ROOT, args.models_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(model_artifact_integrity_markdown(payload), encoding="utf-8")
    print(
        "model artifact integrity: "
        f"verified={payload['verified']}, models={payload['n_models']}, "
        f"npz_arrays={payload['n_npz_arrays']}, joblib_predictors={payload['n_joblib_predictors']}, "
        f"checks={payload['n_checks']}, issues={payload['n_issues']}"
    )
    if args.fail_on_error and not payload["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
