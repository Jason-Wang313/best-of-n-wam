from __future__ import annotations

import numpy as np
import pytest

from wam_inference_value.theorem import binary_best_of_n_finite, utility_best_of_n_finite


def test_binary_utility_recovers_binary_law():
    scores = np.array([0.0, 0.5, 0.5, 1.0])
    success = np.array([0, 1, 0, 1])
    assert utility_best_of_n_finite(scores, success, [1, 2, 8]) == binary_best_of_n_finite(
        scores, success, [1, 2, 8]
    )


def test_constant_utility_returns_constant():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    utilities = np.full(4, 7.25)
    curve = utility_best_of_n_finite(scores, utilities, [1, 2, 16])
    assert all(v == pytest.approx(7.25) for v in curve.values())


def test_oracle_score_monotone_improvement():
    utilities = np.array([-1.0, 0.0, 0.5, 2.0, 4.0])
    scores = utilities.copy()
    curve = utility_best_of_n_finite(scores, utilities, [1, 2, 4, 8])
    vals = [curve[n] for n in [1, 2, 4, 8]]
    assert vals == sorted(vals)
    assert vals[-1] > vals[0]


def test_random_score_gives_little_selection_advantage_on_large_independent_pool():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=4000)
    utilities = rng.normal(size=4000)
    curve = utility_best_of_n_finite(scores, utilities, [1, 64])
    assert abs(curve[64] - curve[1]) < 0.08
