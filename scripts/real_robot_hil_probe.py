from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import results_dir
from wam_inference_value.real_robot_probe import real_robot_hil_probe_markdown, run_real_robot_hil_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--no-hardware", action="store_true", help="Skip local serial/Windows PnP hardware inspection.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "real_robot_hil_probe_report.md")
    args = parser.parse_args()
    payload = run_real_robot_hil_probe(
        ROOT,
        output_results_dir=args.results_dir or results_dir(),
        inspect_hardware=not args.no_hardware,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(real_robot_hil_probe_markdown(payload), encoding="utf-8")
    print(
        "real robot/HIL probe: "
        f"verified={payload.get('verified')}, "
        f"possible_hardware={payload.get('possible_hardware_device_count')}, "
        f"trial_metrics={payload.get('trial_metric_artifact_count')}, "
        f"claim_ready={payload.get('real_robot_or_hil_claim_ready')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
