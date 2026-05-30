from __future__ import annotations

import numpy as np

from wam_inference_value.theorem import binary_best_of_n_finite, utility_best_of_n_finite


def test_tie_group_uses_group_mean_utility():
    scores = np.array([0.0, 1.0, 1.0])
    utility = np.array([0.0, 1.0, 3.0])
    # With N=1, empirical mean. With N=2, top tie group has mean utility 2
    # and is selected unless both samples are the low score.
    out = utility_best_of_n_finite(scores, utility, [1, 2])
    assert np.isclose(out[1], np.mean(utility))
    assert np.isclose(out[2], 2.0 * (1.0 - (1.0 / 3.0) ** 2))


def test_all_success_and_no_success_edges():
    scores = [0.0, 1.0, 2.0]
    assert binary_best_of_n_finite(scores, [1, 1, 1], [1, 8])[1] == 1.0
    assert binary_best_of_n_finite(scores, [1, 1, 1], [1, 8])[8] == 1.0
    assert binary_best_of_n_finite(scores, [0, 0, 0], [1, 8])[1] == 0.0
    assert binary_best_of_n_finite(scores, [0, 0, 0], [1, 8])[8] == 0.0
