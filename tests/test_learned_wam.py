from __future__ import annotations

import numpy as np

from wam_inference_value.envs import BlockPush2D
from wam_inference_value.evaluation import normalize_values
from wam_inference_value.learned_wam import LearnedWamLiteModel, generate_blockpush_dataset
from wam_inference_value.rollouts import make_rollout_pool, sample_action_sequences


def test_learned_wam_trains_and_predicts_finite_metrics():
    env = BlockPush2D()
    train = generate_blockpush_dataset(
        n_states=3,
        rollouts_per_state=12,
        mismatch="none",
        seed=501,
        split="train",
        env=env,
        max_horizon=6,
    )
    model = LearnedWamLiteModel.fit(train, ridge=1e-3, episode_horizon=env.config.episode_horizon)

    state = env.sample_state(777, mismatch="mild")
    actions = sample_action_sequences(env, state, 4, 5, 778)
    metrics = model.predict_batch_metrics(env, state, actions)
    assert len(metrics) == 4
    assert np.all(np.isfinite([m.final_distance for m in metrics]))
    assert np.all(np.isfinite([m.utility for m in metrics]))

    pred = model.predict_next_state_and_utility(env, state, np.array([0.2, 0.1]))
    assert pred.final_obj_xy.shape == (2,)
    assert np.isfinite(pred.utility)


def test_rollout_pool_uses_learned_backend():
    env = BlockPush2D()
    train = generate_blockpush_dataset(
        n_states=2,
        rollouts_per_state=10,
        mismatch="none",
        seed=601,
        split="train",
        env=env,
        max_horizon=6,
    )
    model = LearnedWamLiteModel.fit(train, ridge=1e-3, episode_horizon=env.config.episode_horizon)
    state = env.sample_state(602, mismatch="mild")
    pool = make_rollout_pool(
        env,
        state,
        state_id=0,
        mismatch="mild",
        n_rollouts=8,
        seed=603,
        horizon=5,
        dynamics_backend="learned",
        learned_model=model,
    )
    assert len(pool.records) == 8
    assert np.all(np.isfinite(pool.imagined_utility))
    assert np.all(np.isfinite(pool.real_utility))


def test_normalized_utility_degenerate_pool_is_finite():
    normalized = normalize_values(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(normalized, 0.5)
