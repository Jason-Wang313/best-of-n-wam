from __future__ import annotations

import numpy as np

from wam_inference_value.envs import DrawerPull1D
from wam_inference_value.envs.toy_envs import sample_toy_action_sequences


def test_drawer_deterministic_and_rollout_finite():
    env = DrawerPull1D()
    s1 = env.sample_state(12, "mild")
    s2 = env.sample_state(12, "mild")
    assert np.allclose(s1.vector, s2.vector)
    actions = sample_toy_action_sequences(env, s1, 8, env.horizon, 13)
    metrics = env.rollout_batch_metrics(s1, actions, s1.params)
    assert len(metrics) == 8
    assert np.all(np.isfinite([m.utility for m in metrics]))
