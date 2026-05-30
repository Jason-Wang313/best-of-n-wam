from __future__ import annotations

import numpy as np

from wam_inference_value.envs import DeformableToyEnv
from wam_inference_value.envs.toy_envs import sample_toy_action_sequences


def test_deformable_toy_env_runs_and_penalizes_damage():
    env = DeformableToyEnv()
    state = env.sample_state(41, "severe", state_id=0)
    actions = sample_toy_action_sequences(env, state, 8, env.horizon, 42)
    metrics = env.rollout_batch_metrics(state, actions, state.params)
    assert len(metrics) == 8
    assert all(np.isfinite(m.utility) for m in metrics)
    gentle = np.tile(np.array([[0.2, 0.8]]), (env.horizon, 1))
    harsh = np.tile(np.array([[1.0, 0.0]]), (env.horizon, 1))
    gentle_metrics = env.rollout(state, gentle, state.params)[1]
    harsh_metrics = env.rollout(state, harsh, state.params)[1]
    assert harsh_metrics.safety_violation > gentle_metrics.safety_violation
