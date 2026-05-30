from __future__ import annotations

import numpy as np

from wam_inference_value.evaluation import marginal_greedy_allocate, uniform_allocate


def test_marginal_allocation_prefers_high_marginal_curve():
    strong = np.array([0.2, 0.6, 0.8, 0.9])
    weak = np.array([0.2, 0.25, 0.3, 0.32])
    alloc = marginal_greedy_allocate([strong, weak], total_budget=5)
    assert alloc[0] > alloc[1]
    uniform = uniform_allocate(2, total_budget=5, max_n=4)
    assert sum(alloc) <= 5
    assert sum(uniform) == 5
