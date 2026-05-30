from __future__ import annotations

import numpy as np

from wam_inference_value.envs import BlockPush2D
from wam_inference_value.rollouts import make_rollout_pool, sample_action_sequences


def test_reset_and_step_deterministic_under_seed():
    env = BlockPush2D()
    s1 = env.sample_state(123, mismatch="mild")
    s2 = env.sample_state(123, mismatch="mild")
    assert np.allclose(s1.obj_xy, s2.obj_xy)
    assert np.allclose(s1.target_xy, s2.target_xy)
    action = np.array([0.5, 0.1])
    n1 = env.step(s1, action)
    n2 = env.step(s2, action)
    assert np.allclose(n1.obj_xy, n2.obj_xy)


def test_true_and_imagined_dynamics_can_differ():
    env = BlockPush2D()
    state = env.sample_state(5, mismatch="severe")
    actions = np.tile(np.array([[0.9, 0.0]]), (env.config.horizon, 1))
    _, imagined, _ = env.rollout(state, actions, env.nominal_params)
    _, real, _ = env.rollout(state, actions, state.true_params)
    assert abs(imagined.final_distance - real.final_distance) > 1e-3


def test_rollout_sampler_shapes_and_finite_utilities():
    env = BlockPush2D()
    state = env.sample_state(10, mismatch="mild")
    actions = sample_action_sequences(env, state, 16, env.config.horizon, 11)
    assert actions.shape == (16, env.config.horizon, 2)
    pool = make_rollout_pool(env, state, 0, "mild", 16, 12)
    assert len(pool.records) == 16
    assert np.all(np.isfinite(pool.real_utility))
    assert np.all(np.isfinite(pool.imagined_utility))
