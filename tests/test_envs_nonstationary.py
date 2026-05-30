from __future__ import annotations

import numpy as np

from wam_inference_value.envs import NonstationaryPhysicalShiftEnv


def test_nonstationary_shift_changes_params():
    env = NonstationaryPhysicalShiftEnv()
    state = env.sample_state(31, "nonstationary", state_id=0)
    pre = env.shifted_params(state.params, 0)
    post = env.shifted_params(state.params, env.episode_horizon)
    assert post["friction"] > pre["friction"]
    next_state = env.step(state, np.array([0.5, 0.1]), state.params)
    assert next_state.t == state.t + 1
