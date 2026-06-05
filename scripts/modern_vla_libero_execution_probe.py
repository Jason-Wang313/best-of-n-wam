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
    run_modern_vla_libero_execution_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attempt one-step pretrained SmolVLA execution in LIBERO.")
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--libero-source", type=Path, default=None)
    parser.add_argument("--libero-config", type=Path, default=None)
    parser.add_argument("--model-id", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    python_path = args.python or default_libero_python(ROOT)
    libero_source = args.libero_source or default_libero_source(ROOT)
    libero_config = args.libero_config or default_libero_config(ROOT)
    payload = run_modern_vla_libero_execution_probe(
        ROOT,
        python_path=python_path,
        libero_source=libero_source,
        libero_config=libero_config,
        model_id=args.model_id,
        suite=args.suite,
        task_index=args.task_index,
        seed=args.seed,
        horizon=args.horizon,
        timeout_s=args.timeout_s,
    )
    print(
        "modern VLA LIBERO execution probe: "
        f"verified={payload.get('verified')}, "
        f"policy_loaded={payload.get('policy_loaded')}, "
        f"action_selected={payload.get('action_selected')}, "
        f"libero_step={payload.get('libero_step_succeeded')}, "
        f"failure_stage={payload.get('failure_stage')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
