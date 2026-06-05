from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.modern_vla_execution_probe import (  # noqa: E402
    default_libero_config,
    default_libero_python,
    default_libero_source,
)
from wam_inference_value.modern_vla_policy_eval import run_modern_vla_libero_policy_eval  # noqa: E402


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heldout sparse-success SmolVLA policy eval in LIBERO.")
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--libero-source", type=Path, default=None)
    parser.add_argument("--libero-config", type=Path, default=None)
    parser.add_argument("--model-id", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seeds", type=_parse_seeds, default=[300, 301, 302, 303, 304])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    python_path = args.python or default_libero_python(ROOT)
    libero_source = args.libero_source or default_libero_source(ROOT)
    libero_config = args.libero_config or default_libero_config(ROOT)
    payload = run_modern_vla_libero_policy_eval(
        ROOT,
        python_path=python_path,
        libero_source=libero_source,
        libero_config=libero_config,
        model_id=args.model_id,
        suite=args.suite,
        task_index=args.task_index,
        seeds=args.seeds,
        horizon=args.horizon,
        max_steps=args.max_steps,
        timeout_s=args.timeout_s,
        append_existing=args.append_existing,
    )
    print(
        "modern VLA LIBERO policy eval: "
        f"verified={payload.get('verified')}, "
        f"episodes={payload.get('eval_episodes')}, "
        f"successes={payload.get('eval_successes')}, "
        f"success_rate={payload.get('eval_success_rate')}, "
        f"failure_stage={payload.get('failure_stage')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
