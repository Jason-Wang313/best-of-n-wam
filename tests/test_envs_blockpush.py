from __future__ import annotations

import numpy as np


def test_blockpush_seed_determinism_smoke():
    from wam_inference_value.envs import BlockPush2D

    env = BlockPush2D()
    s1 = env.sample_state(123, mismatch="mild")
    s2 = env.sample_state(123, mismatch="mild")
    assert np.allclose(s1.obj_xy, s2.obj_xy)
    assert np.allclose(env.step(s1, np.array([0.4, 0.2])).obj_xy, env.step(s2, np.array([0.4, 0.2])).obj_xy)
