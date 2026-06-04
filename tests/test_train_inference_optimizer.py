from __future__ import annotations

import csv
import json
from pathlib import Path

from wam_inference_value.train_inference_optimizer import optimize_train_inference


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_optimizer_selects_from_seed_delta_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    tables = results / "tables"
    tables.mkdir(parents=True)
    write_json(
        results / "multi_env_suite.json",
        {
            "model_metrics": [
                {"env": "block_push", "model": "mlp_dynamics_wam", "split": "train", "n_samples": 100},
                {"env": "drawer_pull", "model": "mlp_dynamics_wam", "split": "train", "n_samples": 80},
                {"env": "slippery_grasp", "model": "ensemble_wam", "split": "train", "n_samples": 120},
                {"env": "nonstationary_shift", "model": "horizon_wam", "split": "train", "n_samples": 90},
                {"env": "deformable_toy", "model": "horizon_wam", "split": "train", "n_samples": 70},
            ]
        },
    )

    with (tables / "multi_env_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "env",
                "backend",
                "scorer",
                "seed",
                "mismatch",
                "N",
                "normalized_real_utility",
            ],
        )
        writer.writeheader()
        configs = [
            ("block_push", "mlp_dynamics_wam"),
            ("drawer_pull", "mlp_dynamics_wam"),
            ("slippery_grasp", "ensemble_wam"),
            ("nonstationary_shift", "horizon_wam"),
            ("deformable_toy", "horizon_wam"),
        ]
        for env, backend in configs:
            for seed in range(5):
                for n_value in [1, 2, 8]:
                    writer.writerow(
                        {
                            "env": env,
                            "backend": backend,
                            "scorer": "random",
                            "seed": seed,
                            "mismatch": "mild",
                            "N": n_value,
                            "normalized_real_utility": 0.40 + 0.01 * seed,
                        }
                    )
                    writer.writerow(
                        {
                            "env": env,
                            "backend": backend,
                            "scorer": "predicted_utility",
                            "seed": seed,
                            "mismatch": "mild",
                            "N": n_value,
                            "normalized_real_utility": 0.42 + 0.01 * seed + 0.01 * n_value,
                        }
                    )
                writer.writerow(
                    {
                        "env": env,
                        "backend": backend,
                        "scorer": "safety_penalized",
                        "seed": seed,
                        "mismatch": "mild",
                        "N": 4,
                        "normalized_real_utility": 0.55 + 0.01 * seed,
                    }
                )
                writer.writerow(
                    {
                        "env": env,
                        "backend": backend,
                        "scorer": "random",
                        "seed": seed,
                        "mismatch": "mild",
                        "N": 4,
                        "normalized_real_utility": 0.41 + 0.01 * seed,
                    }
                )

    write_json(
        results / "benchmark_robocasa_multitask_wam.json",
        {
            "verified": True,
            "env_ids": ["robocasa/PickPlaceCounterToCabinet"],
            "model_type": "task_conditioned_ridge_state_action_sequence_wam",
            "train_samples": 144,
            "horizon": 3,
            "n_values": [1, 2, 4, 8],
            "promoted_scorer": "learned_energy_regularized",
            "confidence_intervals": {
                "best_learned_minus_random_N8": {"n": 5, "mean": 0.2, "lo": 0.1, "hi": 0.3}
            },
        },
    )

    payload = optimize_train_inference(tmp_path, results)

    assert payload["verified"] is True
    assert payload["not_a_universal_proof"] is True
    assert payload["choice_dimensions"]["rollout_budget"] is True
    assert payload["choice_dimensions"]["safety_policy"] is True
    selected = {row["environment"]: row for row in payload["selected_by_environment"]}
    assert selected["block_push"]["scorer"] == "safety_penalized"
    assert selected["block_push"]["rollout_budget"] == 4
    assert selected["block_push"]["ci_lo_delta_vs_random"] > 0.0
