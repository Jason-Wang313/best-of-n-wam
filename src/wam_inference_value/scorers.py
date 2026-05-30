"""Rollout score functions.

Scores are intentionally score-agnostic: the theorem only needs the empirical
joint distribution of score and real utility.
"""

from __future__ import annotations

import numpy as np

from wam_inference_value.rollouts import RolloutPool


SCORERS = [
    "predicted_goal_distance",
    "predicted_utility",
    "predicted_success",
    "uncertainty_penalized",
    "safety_penalized",
    "random_score",
    "oracle_real_utility",
]


def scores_for_pool(pool: RolloutPool, scorer: str) -> np.ndarray:
    records = pool.records
    if scorer == "predicted_goal_distance":
        return np.asarray([-r.imagined.final_distance - 0.015 * r.imagined.energy for r in records], dtype=float)
    if scorer == "predicted_utility":
        return np.asarray([r.imagined.utility for r in records], dtype=float)
    if scorer == "predicted_success":
        return np.asarray(
            [
                (2.0 if r.imagined.success else 0.0)
                - r.imagined.final_distance
                - 0.01 * r.imagined.energy
                for r in records
            ],
            dtype=float,
        )
    if scorer == "uncertainty_penalized":
        return np.asarray(
            [
                r.imagined.utility
                - 0.42 * (r.max_action_norm**2)
                - 0.05 * r.imagined.energy
                for r in records
            ],
            dtype=float,
        )
    if scorer == "safety_penalized":
        return np.asarray(
            [r.imagined.utility - 0.55 * r.imagined.safety_violation - 0.18 * r.mean_action_norm for r in records],
            dtype=float,
        )
    if scorer == "random_score":
        return np.asarray([r.random_score for r in records], dtype=float)
    if scorer == "oracle_real_utility":
        return np.asarray([r.real.utility for r in records], dtype=float)
    raise ValueError(f"unknown scorer: {scorer}")
