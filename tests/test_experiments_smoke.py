from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from experiments import adaptive_rollout_allocation
from experiments import auc_vs_moment_hierarchy
from experiments import closed_loop_receding_horizon_eval
from experiments import exact_rollout_law_validation
from experiments import nonstationary_dynamics_extension
from experiments import pilot_to_heldout_prediction
from experiments import real_vs_imagined_utility_gap
from experiments import score_function_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_experiments_write_expected_artifacts():
    exact_rollout_law_validation.run(
        Namespace(states=3, rollouts=32, mc_trials=300, seed=201, mismatch="mild", scorer="predicted_utility")
    )
    auc_vs_moment_hierarchy.run(Namespace(states=3, rollouts=32, seed=202, mismatch="mild", scorer="predicted_utility"))
    pilot_to_heldout_prediction.run(
        Namespace(states=3, rollouts=144, splits=1, seed=203, mismatch="mild", scorer="predicted_utility")
    )
    score_function_comparison.run(Namespace(states=3, rollouts=32, seed=204, mismatch="mild"))
    real_vs_imagined_utility_gap.run(Namespace(states=3, rollouts=32, seed=205, scorer="predicted_utility"))
    adaptive_rollout_allocation.run(
        Namespace(
            states=4,
            rollouts=144,
            pilot_k=32,
            max_n=32,
            seed=206,
            mismatch="mild",
            scorer="predicted_utility",
            mean_budgets=[1, 2, 4, 8, 16, 32],
        )
    )
    closed_loop_receding_horizon_eval.run(Namespace(episodes=1, seed=207, mismatch="mild"))
    nonstationary_dynamics_extension.run(Namespace(episodes=1, rollouts=32, mc_trials=200, seed=208))

    expected = [
        "exp1_exact_rollout_law_validation.json",
        "exp2_auc_vs_moment_hierarchy.json",
        "exp3_pilot_to_heldout_prediction.json",
        "exp4_score_function_comparison.json",
        "exp5_real_vs_imagined_utility_gap.json",
        "exp6_adaptive_rollout_allocation.json",
        "exp7_closed_loop_receding_horizon_eval.json",
        "exp8_nonstationary_dynamics_extension.json",
    ]
    for name in expected:
        assert (ROOT / "results" / name).exists()
