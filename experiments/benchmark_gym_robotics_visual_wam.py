from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmark_visual_wam_lite import action_features, fit_visual_model, image_features, predict_visual_model, save_visual_model
from wam_inference_value.benchmarks.gym_robotics_adapter import GymRoboticsAdapter, is_gym_robotics_available
from wam_inference_value.benchmarks.gym_robotics_rollouts import sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]


def render_state(adapter: GymRoboticsAdapter, state: np.ndarray) -> np.ndarray:
    adapter.set_state(state)
    frame = np.asarray(adapter.env.render())
    if frame.ndim != 3 or frame.shape[2] < 3 or float(np.std(frame)) <= 1e-6:
        raise RuntimeError(f"rendered frame is invalid: shape={frame.shape}, std={float(np.std(frame))}")
    return frame[..., :3]


def dataset_rows(adapter: GymRoboticsAdapter, states: int, rollouts: int, horizon: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for state_id in range(int(states)):
        state = adapter.reset(seed + 4099 * state_id)
        img_feat = image_features(render_state(adapter, state))
        pool = sample_rollout_pool(adapter, state, rollouts, horizon, seed + 100_003 * (state_id + 1))
        actions = pool["actions"]
        records = pool["records"]
        x = np.column_stack([np.repeat(img_feat[None, :], len(actions), axis=0), action_features(actions, horizon)])
        utility = np.asarray([r["utility"] for r in records], dtype=float)
        success = np.asarray([float(r["success"]) for r in records], dtype=float)
        final_distance = np.asarray([r["final_distance"] for r in records], dtype=float)
        xs.append(x)
        ys.append(np.column_stack([utility, success, final_distance]))
    return np.vstack(xs), np.vstack(ys)


def evaluate_env(env_id: str, args: argparse.Namespace) -> dict:
    adapter = GymRoboticsAdapter(env_id=env_id, render_mode="rgb_array", horizon=args.horizon)
    try:
        env_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_id)) % 10_000
        train_x, train_y = dataset_rows(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed + env_offset)
        val_x, val_y = dataset_rows(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 20_000 + env_offset)
        model, model_type = fit_visual_model(train_x, train_y, ridge=args.ridge, seed=args.seed + env_offset)
        val_pred = predict_visual_model(model, val_x)
        utility_mae = float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0])))
        utility_corr = float(np.corrcoef(val_pred[:, 0], val_y[:, 0])[0, 1]) if np.std(val_pred[:, 0]) > 1e-12 and np.std(val_y[:, 0]) > 1e-12 else 0.0
        success_mae = float(np.mean(np.abs(val_pred[:, 1] - val_y[:, 1])))
        distance_mae = float(np.mean(np.abs(val_pred[:, 2] - val_y[:, 2])))

        suffix = "joblib" if model_type == "extra_trees_visual_wam" else "npz"
        model_path = results_dir() / "models" / f"benchmark_gym_robotics_visual_{env_id}_wam.{suffix}"
        save_visual_model(model, model_type, model_path, {"env_id": env_id, "mode": "rgb_frame_action_sequence", "model_type": model_type})

        rows = []
        exact_rows = []
        frame_path = results_dir() / "figures" / f"benchmark_gym_robotics_visual_{env_id}_frame.png"
        wrote_frame = False
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 1231 * state_id)
                frame = render_state(adapter, state)
                if not wrote_frame:
                    iio.imwrite(frame_path, frame)
                    wrote_frame = True
                img_feat = image_features(frame)
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1))
                actions = pool["actions"]
                records = pool["records"]
                x = np.column_stack([np.repeat(img_feat[None, :], len(actions), axis=0), action_features(actions, args.horizon)])
                pred = predict_visual_model(model, x)
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                success = np.asarray([float(r["success"]) for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                norm_utility = normalized_utility(real_utility)
                score_sets = {
                    "random": np.random.default_rng(seed + state_id).normal(size=len(real_utility)),
                    "visual_wam": pred[:, 0],
                    "visual_success_head": pred[:, 1],
                    "low_energy": -energy,
                    "oracle_real_utility": real_utility,
                }
                for scorer, scores in score_sets.items():
                    raw_curve = utility_best_of_n_finite(scores, real_utility, N_VALUES)
                    norm_curve = utility_best_of_n_finite(scores, norm_utility, N_VALUES)
                    succ_curve = utility_best_of_n_finite(scores, success, N_VALUES)
                    for n in N_VALUES:
                        rows.append(
                            {
                                "benchmark": env_id,
                                "seed": int(seed),
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "real_utility": float(raw_curve[n]),
                                "normalized_real_utility": float(norm_curve[n]),
                                "success": float(succ_curve[n]),
                            }
                        )
                for n in N_VALUES:
                    exact = utility_best_of_n_finite(pred[:, 0], real_utility, [n])[n]
                    mc = simulate_best_of_n(pred[:, 0], real_utility, n, args.mc_trials, seed + 17 * n + state_id)
                    exact_rows.append(
                        {
                            "benchmark": env_id,
                            "seed": int(seed),
                            "state_id": int(state_id),
                            "N": int(n),
                            "utility_abs_error": float(abs(exact - mc)),
                        }
                    )
        return {
            "benchmark": env_id,
            "available": True,
            "model_path": str(model_path),
            "frame_path": str(frame_path),
            "model_type": model_type,
            "validation": {
                "utility_mae": utility_mae,
                "utility_corr": utility_corr,
                "success_mae": success_mae,
                "distance_mae": distance_mae,
                "train_samples": int(len(train_x)),
                "validation_samples": int(len(val_x)),
            },
            "rows": rows,
            "exact_rows": exact_rows,
        }
    finally:
        adapter.close()


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    unavailable = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            unavailable.append({"env_id": env_id, "reason": reason})
    if unavailable and len(unavailable) == len(args.env_ids):
        summary = {"experiment": "benchmark_gym_robotics_visual_wam", "attempted": True, "available": False, "unavailable": unavailable}
        write_json(results_dir() / "benchmark_gym_robotics_visual_wam.json", summary)
        return summary

    env_summaries = []
    all_rows = []
    all_exact = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            env_summaries.append({"benchmark": env_id, "available": False, "reason": reason})
            continue
        env_summary = evaluate_env(env_id, args)
        env_summaries.append({k: v for k, v in env_summary.items() if k not in {"rows", "exact_rows"}})
        all_rows.extend(env_summary["rows"])
        all_exact.extend(env_summary["exact_rows"])

    curves = pd.DataFrame(all_rows)
    exact = pd.DataFrame(all_exact)
    curves_path = results_dir() / "tables" / "benchmark_gym_robotics_visual_wam_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_gym_robotics_visual_wam_exact_law.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    agg = curves.groupby(["benchmark", "scorer", "N"], dropna=False)[["success", "real_utility", "normalized_real_utility"]].mean().reset_index()
    agg_path = results_dir() / "tables" / "benchmark_gym_robotics_visual_wam_curves_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    seed_agg = curves.groupby(["seed", "scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    seed_metrics = []
    for seed, sub in seed_agg.groupby("seed"):
        n32 = sub[sub["N"] == 32].set_index("scorer")["normalized_real_utility"]
        seed_metrics.append(
            {
                "seed": int(seed),
                "visual_minus_random_N32": float(n32["visual_wam"] - n32["random"]),
                "oracle_minus_visual_N32": float(n32["oracle_real_utility"] - n32["visual_wam"]),
                "visual_minus_low_energy_N32": float(n32["visual_wam"] - n32["low_energy"]),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_path = results_dir() / "tables" / "benchmark_gym_robotics_visual_wam_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)

    plot_df = agg.groupby(["scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    plt.figure(figsize=(7.5, 4.8))
    for scorer, sub in plot_df.groupby("scorer"):
        if scorer in {"random", "visual_wam", "low_energy", "oracle_real_utility"}:
            plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("normalized real utility")
    plt.title("Gymnasium Robotics RGB WAM-lite")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "benchmark_gym_robotics_visual_wam_curves.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    validation_corrs = [(s.get("validation") or {}).get("utility_corr", 0.0) for s in env_summaries if s.get("available")]
    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    cis = {
        "visual_minus_random_N32": ci95(seed_df["visual_minus_random_N32"].to_numpy()),
        "oracle_minus_visual_N32": ci95(seed_df["oracle_minus_visual_N32"].to_numpy()),
        "visual_minus_low_energy_N32": ci95(seed_df["visual_minus_low_energy_N32"].to_numpy()),
    }
    summary = {
        "experiment": "benchmark_gym_robotics_visual_wam",
        "attempted": True,
        "available": True,
        "verified": bool(
            exact_mae is not None
            and exact_mae < args.max_exact_mae
            and float(np.nanmean(validation_corrs)) > args.min_mean_corr
            and cis["visual_minus_random_N32"]["lo"] > 0.0
        ),
        "benchmark": "Gymnasium Robotics Fetch RGB",
        "mode": "rgb_frame_action_sequence",
        "env_ids": [s["benchmark"] for s in env_summaries if s.get("available")],
        "unavailable": unavailable,
        "n_rollout_pools": int(curves[["benchmark", "seed", "state_id"]].drop_duplicates().shape[0]),
        "n_rollouts": int(args.rollouts),
        "exact_law_utility_mae": exact_mae,
        "mean_validation_utility_corr": float(np.nanmean(validation_corrs)) if validation_corrs else None,
        "model_metrics": env_summaries,
        "confidence_intervals": cis,
        "artifacts": {
            "curves": str(curves_path),
            "aggregate": str(agg_path),
            "exact_law": str(exact_path),
            "seed_metrics": str(seed_path),
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / "benchmark_gym_robotics_visual_wam.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train-states", type=int, default=10)
    parser.add_argument("--train-rollouts", type=int, default=40)
    parser.add_argument("--val-states", type=int, default=4)
    parser.add_argument("--val-rollouts", type=int, default=40)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=40)
    parser.add_argument("--mc-trials", type=int, default=600)
    parser.add_argument("--ridge", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=35001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[35001, 35002, 35003, 35004, 35005])
    parser.add_argument("--min-mean-corr", type=float, default=0.15)
    parser.add_argument("--max-exact-mae", type=float, default=0.08)
    args = parser.parse_args()
    summary = run(args)
    print(
        "benchmark Gymnasium Robotics visual WAM complete: "
        f"available={summary.get('available')}, verified={summary.get('verified')}, "
        f"envs={summary.get('env_ids')}, exact_mae={summary.get('exact_law_utility_mae')}, "
        f"mean_corr={summary.get('mean_validation_utility_corr')}"
    )


if __name__ == "__main__":
    main()
