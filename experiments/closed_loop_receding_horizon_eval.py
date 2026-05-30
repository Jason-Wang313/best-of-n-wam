from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.envs import BlockPush2D, BlockPushConfig
from wam_inference_value.evaluation import (
    N_VALUES,
    add_backend_args,
    backend_suffix,
    ci95,
    ensure_result_dirs,
    load_model_for_backend,
    results_dir,
    seed_list_from_args,
    write_json,
)
from wam_inference_value.rollouts import make_rollout_pool
from wam_inference_value.scorers import scores_for_pool


def closed_loop_scores(env: BlockPush2D, pool, scorer: str, learned_model=None) -> np.ndarray:
    if scorer == "oracle_first_action":
        scores = []
        for record in pool.records:
            next_state = env.step(pool.state, record.actions[0], pool.state.true_params, use_nonstationary_shift=True)
            progress = env.distance_to_target(pool.state) - env.distance_to_target(next_state)
            energy = float(np.dot(record.actions[0], record.actions[0]))
            success = env.distance_to_target(next_state) <= env.config.target_radius
            scores.append(2.5 * float(success) + progress - 0.02 * energy)
        return np.asarray(scores, dtype=float)
    if scorer == "learned_first_action":
        if learned_model is None:
            raise ValueError("learned_first_action scorer requires learned_model")
        first_actions = np.asarray([r.actions[0] for r in pool.records], dtype=float)[:, None, :]
        predicted = learned_model.predict_batch_metrics(env, pool.state, first_actions)
        return np.asarray(
            [
                2.5 * float(m.success) + m.progress - 0.02 * m.energy - 0.25 * m.final_distance
                for m in predicted
            ],
            dtype=float,
        )
    if scorer == "learned_horizon_goal_hybrid":
        state = pool.state
        to_goal = state.target_xy - state.obj_xy
        dist = float(np.linalg.norm(to_goal))
        goal_dir = to_goal / dist if dist > 1e-12 else np.array([1.0, 0.0])
        first_action_progress = np.asarray(
            [float(np.dot(r.actions[0], goal_dir)) - 0.02 * float(np.dot(r.actions[0], r.actions[0])) for r in pool.records],
            dtype=float,
        )
        return scores_for_pool(pool, "predicted_utility") + 0.75 * first_action_progress
    if scorer != "ideal_action":
        return scores_for_pool(pool, scorer)
    state = pool.state
    to_goal = state.target_xy - state.obj_xy
    dist = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / dist if dist > 1e-12 else np.array([1.0, 0.0])
    remaining = max(2.5, env.config.episode_horizon - state.t)
    desired_mag = min(env.config.max_push, max(0.12, dist / remaining) / 0.75)
    ideal_action = goal_dir * desired_mag
    return np.asarray(
        [-float(np.sum((r.actions[0] - ideal_action) ** 2)) - 0.005 * r.imagined.energy for r in pool.records],
        dtype=float,
    )


def run_episode(
    env: BlockPush2D,
    seed: int,
    N: int,
    scorer: str,
    mismatch: str,
    dynamics_backend: str,
    learned_model,
) -> dict:
    state = env.sample_state(seed, mismatch=mismatch, state_id=seed % 10_000)
    initial_distance = env.distance_to_target(state)
    total_energy = 0.0
    for step in range(env.config.episode_horizon):
        pool = make_rollout_pool(
            env,
            state,
            state_id=step,
            mismatch=mismatch,
            n_rollouts=N,
            seed=seed + 10_007 * (step + 1),
            horizon=env.config.horizon,
            dynamics_backend=dynamics_backend,
            learned_model=learned_model,
        )
        scores = closed_loop_scores(env, pool, scorer, learned_model)
        best = int(np.argmax(scores))
        action = pool.records[best].actions[0]
        total_energy += float(np.dot(action, action))
        state = env.step(state, action, state.true_params, use_nonstationary_shift=True)
        if env.distance_to_target(state) <= env.config.target_radius:
            break
    final_distance = env.distance_to_target(state)
    success = final_distance <= env.config.target_radius
    utility = (
        env.config.success_bonus * float(success)
        + env.config.progress_weight * (initial_distance - final_distance)
        - env.config.distance_weight * final_distance
        - env.config.energy_weight * total_energy
    )
    return {
        "success": float(success),
        "final_distance": float(final_distance),
        "utility": float(utility),
        "steps": step + 1,
        "compute_rollouts": int((step + 1) * N),
    }


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    env = BlockPush2D(BlockPushConfig(horizon=5, episode_horizon=5, target_radius=0.25))
    dynamics_backend = getattr(args, "dynamics_backend", "analytic")
    learned_model = load_model_for_backend(args)
    seeds = seed_list_from_args(args)
    rows = []
    useful_scorer = "ideal_action" if dynamics_backend == "analytic" else "learned_horizon_goal_hybrid"
    scorers = getattr(args, "scorers", None) or [useful_scorer, "random_score", "oracle_first_action"]
    for seed_id, seed in enumerate(seeds):
        for scorer in scorers:
            for N in N_VALUES:
                for ep in range(args.episodes):
                    rec = run_episode(env, seed + ep * 37, N, scorer, args.mismatch, dynamics_backend, learned_model)
                    rec.update(
                        {
                            "seed": int(seed),
                            "seed_id": int(seed_id),
                            "episode": ep,
                            "N": N,
                            "scorer": scorer,
                            "mismatch": args.mismatch,
                        }
                    )
                    rows.append(rec)
    df = pd.DataFrame(rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"exp7_closed_loop_receding_horizon_eval{suffix}.csv"
    df.to_csv(table_path, index=False)
    agg = df.groupby(["scorer", "N"]).agg(
        success=("success", "mean"),
        utility=("utility", "mean"),
        final_distance=("final_distance", "mean"),
        compute_rollouts=("compute_rollouts", "mean"),
    ).reset_index()
    agg_path = results_dir() / "tables" / f"exp7_closed_loop_receding_horizon_aggregate{suffix}.csv"
    agg.to_csv(agg_path, index=False)

    plt.figure(figsize=(7.4, 4.7))
    for scorer, sub in agg.groupby("scorer"):
        plt.plot(sub["N"], sub["success"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("rollouts per receding-horizon decision")
    plt.ylabel("closed-loop success")
    plt.title("Closed-loop receding-horizon planning benefits from useful scores")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"exp7_closed_loop_success{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    useful = agg[agg["scorer"] == useful_scorer].set_index("N")
    rand = agg[agg["scorer"] == "random_score"].set_index("N")
    oracle = agg[agg["scorer"] == "oracle_first_action"].set_index("N")
    seed_metrics = []
    seed_agg = df.groupby(["seed", "scorer", "N"], dropna=False).agg(
        success=("success", "mean"),
        utility=("utility", "mean"),
    ).reset_index()
    for seed, sub in seed_agg.groupby("seed"):
        useful_seed = sub[sub["scorer"] == useful_scorer].set_index("N")
        rand_seed = sub[sub["scorer"] == "random_score"].set_index("N")
        oracle_seed = sub[sub["scorer"] == "oracle_first_action"].set_index("N")
        seed_metrics.append(
            {
                "seed": int(seed),
                "useful_success_gain_N64_minus_N1": float(useful_seed.loc[64, "success"] - useful_seed.loc[1, "success"]),
                "useful_utility_gain_N64_minus_N1": float(useful_seed.loc[64, "utility"] - useful_seed.loc[1, "utility"]),
                "useful_minus_random_success_N64": float(useful_seed.loc[64, "success"] - rand_seed.loc[64, "success"]),
                "oracle_first_action_minus_useful_success_N64": float(
                    oracle_seed.loc[64, "success"] - useful_seed.loc[64, "success"]
                ),
            }
        )
    seed_metric_df = pd.DataFrame(seed_metrics)
    summary = {
        "experiment": "closed_loop_receding_horizon_eval",
        "dynamics_backend": dynamics_backend,
        "episodes_per_setting": args.episodes,
        "mismatch": args.mismatch,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "useful_scorer": useful_scorer,
        "useful_success_gain_N64_minus_N1": float(useful.loc[64, "success"] - useful.loc[1, "success"]),
        "useful_utility_gain_N64_minus_N1": float(useful.loc[64, "utility"] - useful.loc[1, "utility"]),
        "useful_minus_random_success_N64": float(useful.loc[64, "success"] - rand.loc[64, "success"]),
        "oracle_first_action_minus_useful_success_N64": float(oracle.loc[64, "success"] - useful.loc[64, "success"]),
        "confidence_intervals": {
            "useful_success_gain_N64_minus_N1": ci95(seed_metric_df["useful_success_gain_N64_minus_N1"].to_numpy()),
            "useful_utility_gain_N64_minus_N1": ci95(seed_metric_df["useful_utility_gain_N64_minus_N1"].to_numpy()),
            "useful_minus_random_success_N64": ci95(seed_metric_df["useful_minus_random_success_N64"].to_numpy()),
            "oracle_first_action_minus_useful_success_N64": ci95(
                seed_metric_df["oracle_first_action_minus_useful_success_N64"].to_numpy()
            ),
        },
        "aggregate": agg.to_dict(orient="records"),
        "artifacts": {"table": str(table_path), "aggregate": str(agg_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / f"exp7_closed_loop_receding_horizon_eval{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--mismatch", type=str, default="none")
    add_backend_args(parser)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp7 complete: "
        f"useful N64-N1 success delta={summary['useful_success_gain_N64_minus_N1']:.3f}, "
        f"useful-random N64 delta={summary['useful_minus_random_success_N64']:.3f}"
    )


if __name__ == "__main__":
    main()
