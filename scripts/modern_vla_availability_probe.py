from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import results_dir
from wam_inference_value.modern_vla_probe import modern_vla_availability_markdown, run_modern_vla_availability_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--no-hf", action="store_true", help="Skip public Hugging Face metadata probes.")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "modern_vla_availability_probe_report.md")
    args = parser.parse_args()
    payload = run_modern_vla_availability_probe(
        ROOT,
        output_results_dir=args.results_dir or results_dir(),
        probe_hf=not args.no_hf,
        timeout_s=args.timeout_s,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(modern_vla_availability_markdown(payload), encoding="utf-8")
    print(
        "modern VLA availability probe: "
        f"verified={payload.get('verified')}, "
        f"vla_package_importable={payload.get('vla_package_importable')}, "
        f"local_vla_like={payload.get('local_vla_like_count')}, "
        f"hf_reachable={payload.get('hf_reachable_count')}, "
        f"ready_for_policy_eval={payload.get('ready_for_policy_eval')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
