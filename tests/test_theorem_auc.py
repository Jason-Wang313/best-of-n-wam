from __future__ import annotations

import numpy as np

from wam_inference_value.theorem import auc_kappa, binary_best_of_n_finite, n2_auc_identity


def test_n2_auc_identity_hard_assertion():
    scores = np.array([0.1, 0.2, 0.2, 0.8, 0.9, 1.0])
    success = np.array([0, 0, 1, 0, 1, 1], dtype=float)
    p = float(np.mean(success))
    kappa = auc_kappa(scores, success)
    exact = binary_best_of_n_finite(scores, success, [2])[2]
    assert exact == n2_auc_identity(p, kappa)
