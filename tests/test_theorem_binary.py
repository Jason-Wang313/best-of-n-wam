from __future__ import annotations

import itertools

import numpy as np
import pytest

from wam_inference_value.theorem import (
    auc_kappa,
    binary_best_of_n_finite,
    n2_auc_identity,
    simulate_best_of_n,
)


def brute_force(scores, values, N):
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    total = 0.0
    count = 0
    for tup in itertools.product(range(len(scores)), repeat=N):
        chosen_scores = scores[list(tup)]
        max_score = np.max(chosen_scores)
        tied_positions = np.flatnonzero(chosen_scores == max_score)
        total += float(np.mean(values[[tup[i] for i in tied_positions]]))
        count += 1
    return total / count


def test_finite_law_matches_brute_force_with_ties():
    scores = np.array([0.1, 0.2, 0.2, 0.8])
    success = np.array([0, 1, 0, 1])
    for N in [1, 2, 3, 4]:
        exact = binary_best_of_n_finite(scores, success, [N])[N]
        assert exact == pytest.approx(brute_force(scores, success, N))


def test_monte_carlo_matches_within_tolerance():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    success = np.array([0, 0, 1, 1])
    exact = binary_best_of_n_finite(scores, success, [4])[4]
    mc = simulate_best_of_n(scores, success, 4, n_trials=30_000, seed=1)
    assert abs(exact - mc) < 0.015


def test_n1_equals_p():
    scores = np.array([3.0, 1.0, 2.0, 4.0])
    success = np.array([1, 0, 0, 1])
    assert binary_best_of_n_finite(scores, success, [1])[1] == np.mean(success)


def test_n2_auc_identity_matches_tie_aware_law():
    scores = np.array([0.0, 1.0, 1.0, 3.0, 3.0])
    success = np.array([0, 1, 0, 1, 0])
    p = float(np.mean(success))
    kappa = auc_kappa(scores, success)
    f2 = binary_best_of_n_finite(scores, success, [2])[2]
    assert f2 == pytest.approx(n2_auc_identity(p, kappa))


def test_all_correct_and_no_correct_edges():
    scores = np.array([0.0, 1.0, 2.0])
    assert binary_best_of_n_finite(scores, np.ones(3), [1, 2, 8]) == {1: 1.0, 2: 1.0, 8: 1.0}
    assert binary_best_of_n_finite(scores, np.zeros(3), [1, 2, 8]) == {1: 0.0, 2: 0.0, 8: 0.0}
