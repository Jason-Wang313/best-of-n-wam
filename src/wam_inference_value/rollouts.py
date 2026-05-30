"""Rollout sampling and pool construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wam_inference_value.envs.block_push_2d import (
    BlockPush2D,
    BlockPushState,
    RolloutMetrics,
)


@dataclass(frozen=True)
class RolloutRecord:
    actions: np.ndarray
    imagined: RolloutMetrics
    real: RolloutMetrics
    random_score: float
    mean_action_norm: float
    max_action_norm: float


@dataclass(frozen=True)
class RolloutPool:
    state_id: int
    mismatch: str
    state: BlockPushState
    records: tuple[RolloutRecord, ...]

    @property
    def real_success(self) -> np.ndarray:
        return np.asarray([float(r.real.success) for r in self.records], dtype=float)

    @property
    def real_utility(self) -> np.ndarray:
        return np.asarray([r.real.utility for r in self.records], dtype=float)

    @property
    def imagined_utility(self) -> np.ndarray:
        return np.asarray([r.imagined.utility for r in self.records], dtype=float)


def sample_action_sequences(
    env: BlockPush2D,
    state: BlockPushState,
    n_rollouts: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Sample random-shooting plus noisy goal-directed action sequences."""

    rng = np.random.default_rng(seed)
    out = np.zeros((int(n_rollouts), int(horizon), 2), dtype=float)
    to_goal = state.target_xy - state.obj_xy
    goal_norm = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / goal_norm if goal_norm > 1e-12 else np.array([1.0, 0.0])
    perp = np.array([-goal_dir[1], goal_dir[0]])

    for i in range(int(n_rollouts)):
        mode = rng.choice(["goal", "cautious", "explore", "burst"], p=[0.48, 0.20, 0.20, 0.12])
        if mode == "goal":
            base_mag = rng.uniform(0.45, 0.88)
            noise_scale = rng.uniform(0.05, 0.20)
        elif mode == "cautious":
            base_mag = rng.uniform(0.18, 0.48)
            noise_scale = rng.uniform(0.02, 0.12)
        elif mode == "burst":
            base_mag = rng.uniform(0.86, 1.0)
            noise_scale = rng.uniform(0.01, 0.10)
        else:
            base_mag = rng.uniform(0.15, 1.0)
            noise_scale = rng.uniform(0.15, 0.55)

        for t in range(int(horizon)):
            if mode == "explore" and rng.random() < 0.65:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                lateral = rng.normal(0.0, noise_scale)
                forward = max(0.0, rng.normal(1.0, noise_scale))
                direction = forward * goal_dir + lateral * perp
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1e-12 else goal_dir
            decay = 1.0 - 0.45 * (t / max(1, horizon - 1))
            mag = np.clip(base_mag * decay + rng.normal(0.0, 0.06), 0.0, env.config.max_push)
            out[i, t] = direction * mag
    return out


def make_rollout_pool(
    env: BlockPush2D,
    state: BlockPushState,
    state_id: int,
    mismatch: str,
    n_rollouts: int,
    seed: int,
    horizon: int | None = None,
    dynamics_backend: str = "analytic",
    learned_model: Any | None = None,
) -> RolloutPool:
    horizon = env.config.horizon if horizon is None else int(horizon)
    actions = sample_action_sequences(env, state, n_rollouts, horizon, seed)
    records: list[RolloutRecord] = []
    rng = np.random.default_rng(seed + 10_000)
    real_metrics = env.rollout_batch_metrics(state, actions, state.true_params, use_nonstationary_shift=True)
    if dynamics_backend == "analytic":
        imagined_metrics = env.rollout_batch_metrics(state, actions, env.nominal_params, use_nonstationary_shift=False)
    elif dynamics_backend == "learned":
        if learned_model is None:
            raise ValueError("dynamics_backend='learned' requires learned_model")
        imagined_metrics = learned_model.predict_batch_metrics(env, state, actions)
    elif dynamics_backend == "oracle_true":
        imagined_metrics = real_metrics
    else:
        raise ValueError(f"unknown dynamics backend: {dynamics_backend}")
    for seq, imagined, real in zip(actions, imagined_metrics, real_metrics):
        norms = np.linalg.norm(seq, axis=1)
        records.append(
            RolloutRecord(
                actions=seq,
                imagined=imagined,
                real=real,
                random_score=float(rng.normal()),
                mean_action_norm=float(np.mean(norms)),
                max_action_norm=float(np.max(norms)),
            )
        )
    return RolloutPool(state_id=int(state_id), mismatch=mismatch, state=state, records=tuple(records))


def generate_rollout_pools(
    n_states: int,
    n_rollouts: int,
    mismatch: str,
    seed: int,
    horizon: int | None = None,
    env: BlockPush2D | None = None,
    dynamics_backend: str = "analytic",
    learned_model: Any | None = None,
) -> list[RolloutPool]:
    env = env or BlockPush2D()
    pools = []
    for state_id in range(int(n_states)):
        state_seed = int(seed + 7919 * state_id + 17)
        state = env.sample_state(state_seed, mismatch=mismatch, state_id=state_id)
        pool = make_rollout_pool(
            env=env,
            state=state,
            state_id=state_id,
            mismatch=mismatch,
            n_rollouts=n_rollouts,
            seed=seed + 104_729 * (state_id + 1),
            horizon=horizon,
            dynamics_backend=dynamics_backend,
            learned_model=learned_model,
        )
        pools.append(pool)
    return pools
