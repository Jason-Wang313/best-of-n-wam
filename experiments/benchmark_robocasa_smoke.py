from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.robocasa_adapter import (
    RoboCasaAdapter,
    RoboCasaUnavailableError,
    is_robocasa_available,
)
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


DEFAULT_N_VALUES = [1, 2, 4, 8]


def _write_report(summary: dict) -> None:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not summary.get("available"):
        lines = [
            "# RoboCasa Smoke Report",
            "",
            "- status: unavailable",
            f"- reason: {summary.get('reason')}",
            "",
            "RoboCasa is optional. Full validation requires a separate RoboCasa-compatible Python environment and the official kitchen assets.",
        ]
    else:
        ci = (summary.get("confidence_intervals") or {}).get("distance_progress_minus_random_N8") or {}
        lines = [
            "# RoboCasa Smoke Report",
            "",
            "- status: verified smoke",
            f"- env: `{summary.get('env_id')}`",
            f"- split: `{summary.get('split')}`",
            f"- rollout pools: `{summary.get('n_rollout_pools')}`",
            f"- total rollouts: `{summary.get('n_rollouts_total')}`",
            f"- exact-law utility MAE: `{summary.get('exact_law_utility_mae')}`",
            f"- distance-progress minus random at N8 mean: `{ci.get('mean')}`",
            "",
            "This is a contact-rich RoboCasa kitchen reset/rollout smoke artifact. It is not a full RoboCasa learned-WAM benchmark or closed-loop validation.",
        ]
    (report_dir / "robocasa_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unavailable(reason: str) -> dict:
    ensure_result_dirs()
    summary = {
        "experiment": "benchmark_robocasa_smoke",
        "attempted": True,
        "available": False,
        "verified": False,
        "reason": reason,
    }
    write_json(results_dir() / "benchmark_robocasa_smoke.json", summary)
    _write_report(summary)
    return summary


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    ok, reason = is_robocasa_available()
    if not ok:
        return _unavailable(reason)

    n_values = [int(n) for n in args.n_values]
    if any(n < 1 for n in n_values):
        raise ValueError("all N values must be >= 1")

    all_rows: list[dict] = []
    exact_rows: list[dict] = []
    pool_rows: list[dict] = []
    closed_rows: list[dict] = []
    adapter: RoboCasaAdapter | None = None
    try:
        adapter = RoboCasaAdapter(
            env_id=args.env_id,
            split=args.split,
            horizon=args.horizon,
            camera_width=args.camera_size,
            camera_height=args.camera_size,
            success_bonus=args.success_bonus,
            energy_penalty=args.energy_penalty,
        )
        for state_id in range(args.states):
            seed = int(args.seed + 10_007 * state_id)
            state = adapter.reset_task(seed=seed)
            pool = adapter.sample_rollouts(
                initial_state=state,
                n_rollouts=args.rollouts,
                horizon=args.horizon,
                seed=seed + 17,
            )
            scores = adapter.score_rollouts(pool, seed=seed + 31)
            records = pool["records"]
            real_utility = np.asarray([r["utility"] for r in records], dtype=float)
            norm_utility = normalized_utility(real_utility)
            for r in records:
                rr = dict(r)
                rr.update({"state_id": int(state_id), "seed": seed, "env_id": args.env_id})
                pool_rows.append(rr)
            for scorer, score in scores.items():
                raw_curve = utility_best_of_n_finite(score, real_utility, n_values)
                norm_curve = utility_best_of_n_finite(score, norm_utility, n_values)
                for n in n_values:
                    all_rows.append(
                        {
                            "env_id": args.env_id,
                            "seed": seed,
                            "state_id": int(state_id),
                            "scorer": scorer,
                            "N": int(n),
                            "real_utility": float(raw_curve[n]),
                            "normalized_real_utility": float(norm_curve[n]),
                        }
                    )
                    if scorer in {"distance_progress", "oracle_real_utility"}:
                        mc = simulate_best_of_n(score, real_utility, n, args.mc_trials, seed + 100 * n)
                        exact_rows.append(
                            {
                                "env_id": args.env_id,
                                "seed": seed,
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "utility_exact": float(raw_curve[n]),
                                "utility_mc": float(mc),
                                "utility_abs_error": float(abs(raw_curve[n] - mc)),
                            }
                        )
        if args.closed_loop:
            for scorer in ("random", "distance_progress", "oracle_real_utility"):
                for n in (1, min(8, max(n_values))):
                    adapter.reset_task(seed=args.seed + 55_003 + 17 * n)
                    rec = adapter.run_closed_loop(
                        n=n,
                        steps=args.closed_loop_steps,
                        candidate_horizon=args.horizon,
                        scorer=scorer,
                        seed=args.seed + 99_991 + 37 * n,
                    )
                    rec.update({"env_id": args.env_id, "seed": args.seed, "scorer": scorer, "N": int(n)})
                    closed_rows.append(rec)
    except RoboCasaUnavailableError as exc:
        return _unavailable(str(exc))
    finally:
        if adapter is not None:
            adapter.close()

    curves = pd.DataFrame(all_rows)
    exact = pd.DataFrame(exact_rows)
    pools = pd.DataFrame(pool_rows)
    closed = pd.DataFrame(closed_rows)
    curves_path = results_dir() / "tables" / "benchmark_robocasa_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_robocasa_exact_law.csv"
    pools_path = results_dir() / "tables" / "benchmark_robocasa_rollouts.csv"
    closed_path = results_dir() / "tables" / "benchmark_robocasa_closed_loop.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    pools.to_csv(pools_path, index=False)
    if not closed.empty:
        closed.to_csv(closed_path, index=False)

    max_n = max(n_values)
    seed_metrics = []
    for seed, sub in curves[curves["N"] == max_n].groupby("seed"):
        by_scorer = sub.groupby("scorer")["normalized_real_utility"].mean()
        seed_metrics.append(
            {
                "seed": int(seed),
                f"distance_progress_minus_random_N{max_n}": float(by_scorer.get("distance_progress", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_random_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_distance_progress_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("distance_progress", np.nan)),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_df.to_csv(results_dir() / "tables" / "benchmark_robocasa_seed_metrics.csv", index=False)

    confidence_intervals = {
        key: ci95(seed_df[key].to_numpy())
        for key in seed_df.columns
        if key != "seed"
    }
    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    summary = {
        "experiment": "benchmark_robocasa_smoke",
        "attempted": True,
        "available": True,
        "verified": exact_mae is not None and exact_mae < args.max_exact_mae,
        "env_id": args.env_id,
        "split": args.split,
        "states": int(args.states),
        "rollouts": int(args.rollouts),
        "horizon": int(args.horizon),
        "n_rollout_pools": int(args.states),
        "n_rollouts_total": int(len(pool_rows)),
        "n_values": n_values,
        "exact_law_utility_mae": exact_mae,
        "confidence_intervals": confidence_intervals,
        "curves_path": str(curves_path),
        "exact_path": str(exact_path),
        "rollouts_path": str(pools_path),
        "closed_loop_path": str(closed_path) if not closed.empty else None,
        "note": "RoboCasa smoke only: external dependency and kitchen assets are optional; this is not full learned-WAM RoboCasa validation.",
    }
    write_json(results_dir() / "benchmark_robocasa_smoke.json", summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--states", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--camera-size", type=int, default=16)
    parser.add_argument("--mc-trials", type=int, default=2000)
    parser.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--energy-penalty", type=float, default=0.01)
    parser.add_argument("--max-exact-mae", type=float, default=0.08)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--closed-loop-steps", type=int, default=2)
    args = parser.parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
