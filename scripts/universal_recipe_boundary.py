from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import results_dir
from wam_inference_value.universal_recipe_boundary import (
    build_universal_recipe_boundary,
    universal_recipe_boundary_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "universal_recipe_boundary_report.md")
    args = parser.parse_args()
    payload = build_universal_recipe_boundary(ROOT, args.results_dir or results_dir())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(universal_recipe_boundary_markdown(payload), encoding="utf-8")
    print(
        "universal recipe boundary: "
        f"verified={payload.get('verified')}, "
        f"checks={payload.get('n_checks')}, "
        f"issues={payload.get('n_issues')}, "
        f"worst_case_regret_lb={payload.get('randomized_worst_case_regret_lower_bound')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
