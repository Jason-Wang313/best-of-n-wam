from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json
from wam_inference_value.learned_wam import train_learned_wam_lite


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    model_path = Path(args.model_path) if args.model_path else results_dir() / "models" / "learned_wam_lite.npz"
    model, summary = train_learned_wam_lite(
        model_path=model_path,
        dataset_dir=results_dir() / "datasets",
        id_mismatch=args.id_mismatch,
        seed=args.seed,
        train_states=args.train_states,
        train_rollouts=args.train_rollouts,
        val_states=args.val_states,
        val_rollouts=args.val_rollouts,
        ood_states=args.ood_states,
        ood_rollouts=args.ood_rollouts,
        max_horizon=args.max_horizon,
        ridge=args.ridge,
        ood_mismatches=tuple(args.ood_mismatches),
    )
    del model

    summary_path = results_dir() / "learned_wam_lite_training.json"
    write_json(summary_path, summary)

    dataset_rows = summary["datasets"]
    dataset_path = results_dir() / "tables" / "learned_wam_lite_dataset_summary.csv"
    pd.DataFrame(dataset_rows).to_csv(dataset_path, index=False)

    metric_rows = [summary["metrics"]["train"], summary["metrics"]["validation"], *summary["metrics"]["ood"]]
    metric_path = results_dir() / "tables" / "learned_wam_lite_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)

    summary["artifacts"] = {
        "model": str(model_path),
        "summary": str(summary_path),
        "dataset_summary": str(dataset_path),
        "metrics": str(metric_path),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--id-mismatch", type=str, default="mild")
    parser.add_argument("--train-states", type=int, default=64)
    parser.add_argument("--train-rollouts", type=int, default=96)
    parser.add_argument("--val-states", type=int, default=24)
    parser.add_argument("--val-rollouts", type=int, default=96)
    parser.add_argument("--ood-states", type=int, default=16)
    parser.add_argument("--ood-rollouts", type=int, default=64)
    parser.add_argument("--max-horizon", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--ood-mismatches", nargs="*", default=["severe", "stuck_slip", "nonstationary"])
    args = parser.parse_args()
    summary = run(args)
    val = summary["metrics"]["validation"]
    print(
        "learned WAM-lite train complete: "
        f"validation utility MAE={val['utility_mae']:.4f}, "
        f"final position L2 MAE={val['final_position_l2_mae']:.4f}"
    )


if __name__ == "__main__":
    main()
