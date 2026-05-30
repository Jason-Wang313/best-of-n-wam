from __future__ import annotations

from typing import Any

import numpy as np


def benchmark_feature(state: np.ndarray, actions: np.ndarray, max_horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    if actions.ndim == 2:
        actions = actions[None, :, :]
    n, horizon, action_dim = actions.shape
    padded = np.zeros((n, int(max_horizon), action_dim), dtype=float)
    padded[:, :horizon, :] = actions
    norms = np.linalg.norm(actions, axis=2)
    sums = actions.sum(axis=1)
    return np.column_stack(
        [
            np.repeat(np.asarray(state, dtype=float)[None, :], n, axis=0),
            padded.reshape(n, int(max_horizon) * action_dim),
            sums,
            norms.mean(axis=1),
            norms.max(axis=1),
            np.sum(actions * actions, axis=(1, 2)),
            np.full(n, horizon / max(1, int(max_horizon))),
        ]
    )


def sample_action_sequences(adapter: Any, n_rollouts: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray(adapter.env.action_space.low, dtype=float).reshape(-1)
    high = np.asarray(adapter.env.action_space.high, dtype=float).reshape(-1)
    actions = rng.normal(0.0, 0.55, size=(int(n_rollouts), int(horizon), adapter.action_dim))
    # Mix in occasional stronger shooting rollouts so best-of-N has a real upper tail.
    burst = rng.random(size=(int(n_rollouts), 1, 1)) < 0.25
    actions = np.where(burst, actions * 1.8, actions)
    return np.clip(actions, low, high)


def rollout_sequence(adapter: Any, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    adapter.set_state(state)
    initial_distance = adapter.distance_to_goal()
    energy = 0.0
    total_reward = 0.0
    for action in np.asarray(actions, dtype=float):
        energy += float(np.sum(np.asarray(action, dtype=float) ** 2))
        _, reward, terminated, truncated, _ = adapter.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    final_state = adapter.get_state()
    final_distance = adapter.distance_to_goal()
    success = final_distance <= adapter.success_threshold
    progress = initial_distance - final_distance
    utility = 2.5 * float(success) + progress - final_distance - 0.01 * energy
    return {
        "initial_distance": float(initial_distance),
        "final_distance": float(final_distance),
        "progress": float(progress),
        "energy": float(energy),
        "success": bool(success),
        "utility": float(utility),
        "total_reward": float(total_reward),
        "final_state": final_state,
    }


def sample_rollout_pool(adapter: Any, state: np.ndarray, n_rollouts: int, horizon: int, seed: int) -> dict[str, Any]:
    actions = sample_action_sequences(adapter, n_rollouts, horizon, seed)
    records = []
    for seq in actions:
        rec = rollout_sequence(adapter, state, seq)
        rec["actions"] = seq
        records.append(rec)
    return {"state": np.asarray(state, dtype=float), "actions": actions, "records": records}


def run_closed_loop(
    adapter: Any,
    model: Any,
    scorer: str,
    n: int,
    seed: int,
    steps: int = 15,
    candidate_horizon: int = 12,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = adapter.reset(seed)
    initial_distance = adapter.distance_to_goal()
    energy = 0.0
    for t in range(int(steps)):
        actions = sample_action_sequences(adapter, int(n), int(candidate_horizon), seed + 7919 * (t + 1))
        if scorer == "random":
            scores = rng.normal(size=int(n))
        elif scorer == "oracle":
            scores = np.asarray([rollout_sequence(adapter, state, seq)["utility"] for seq in actions], dtype=float)
        elif scorer == "learned":
            feature_state = adapter.feature_state(state) if hasattr(adapter, "feature_state") else state
            x = benchmark_feature(feature_state, actions, int(feature_horizon or candidate_horizon))
            scores = np.asarray(model.predict(x)[:, -1], dtype=float)
        else:
            raise ValueError(f"unknown closed-loop scorer: {scorer}")
        best = int(np.argmax(scores))
        first_action = actions[best, 0]
        energy += float(np.sum(first_action * first_action))
        adapter.set_state(state)
        adapter.step(first_action)
        state = adapter.get_state()
        if adapter.evaluate_success():
            break
    final_distance = adapter.distance_to_goal()
    success = final_distance <= adapter.success_threshold
    utility = 2.5 * float(success) + (initial_distance - final_distance) - final_distance - 0.01 * energy
    return {
        "success": float(success),
        "utility": float(utility),
        "final_distance": float(final_distance),
        "energy": float(energy),
        "steps": int(t + 1),
        "compute_rollouts": int((t + 1) * n),
    }
