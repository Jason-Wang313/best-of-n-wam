from __future__ import annotations

import numpy as np

from wam_inference_value.envs import SlipperyGrasp1D
from wam_inference_value.envs.toy_envs import sample_toy_action_sequences


def test_grasp_deterministic_and_safety_metric():
    env = SlipperyGrasp1D()
    state = env.sample_state(22, "severe")
    actions = sample_toy_action_sequences(env, state, 8, env.horizon, 23)
    metrics = env.rollout_batch_metrics(state, actions, state.params)
    assert np.all(np.isfinite([m.utility for m in metrics]))
    assert any(m.safety_violation > 0.0 for m in metrics)
