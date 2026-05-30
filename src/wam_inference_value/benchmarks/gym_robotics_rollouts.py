from __future__ import annotations

from typing import Any

import numpy as np

from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature


def sample_action_sequences(adapter: Any, n_rollouts: int, horizon: int, seed: int, state: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray(adapter.env.action_space.low, dtype=float).reshape(-1)
    high = np.asarray(adapter.env.action_space.high, dtype=float).reshape(-1)
    actions = rng.normal(0.0, 0.45, size=(int(n_rollouts), int(horizon), adapter.action_dim))
    actions = np.clip(actions, low, high)

    # Seed a small fraction of the shooting pool with a simple goal-directed
    # controller. The pool is still stochastic, but Fetch tasks otherwise waste
    # many samples on motion that never contacts the object.
    n_guided = max(1, int(0.25 * int(n_rollouts)))
    for i in range(n_guided):
        if state is not None:
            adapter.set_state(state)
        guided = []
        for _ in range(int(horizon)):
            base = adapter.heuristic_action(scale=float(rng.uniform(2.0, 5.0)))
            guided.append(np.clip(base + rng.normal(0.0, 0.12, size=adapter.action_dim), low, high))
        actions[i] = np.asarray(guided, dtype=float)
    rng.shuffle(actions, axis=0)
    return actions


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
    utility = 3.0 * float(success) + 1.5 * progress - final_distance - 0.006 * energy
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
    actions = sample_action_sequences(adapter, n_rollouts, horizon, seed, state)
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
    steps: int = 10,
    candidate_horizon: int = 8,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = adapter.reset(seed)
    initial_distance = adapter.distance_to_goal()
    energy = 0.0
    t = 0
    for t in range(int(steps)):
        actions = sample_action_sequences(adapter, int(n), int(candidate_horizon), seed + 7919 * (t + 1), state)
        if scorer == "random":
            scores = rng.normal(size=int(n))
        elif scorer == "oracle":
            scores = np.asarray([rollout_sequence(adapter, state, seq)["utility"] for seq in actions], dtype=float)
        elif scorer == "learned":
            feature_state = adapter.feature_state(state) if hasattr(adapter, "feature_state") else state
            x = benchmark_feature(feature_state, actions, int(feature_horizon or candidate_horizon))
            scores = np.asarray(model.predict(x)[:, 0], dtype=float)
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
    utility = 3.0 * float(success) + 1.5 * (initial_distance - final_distance) - final_distance - 0.006 * energy
    return {
        "success": float(success),
        "utility": float(utility),
        "final_distance": float(final_distance),
        "energy": float(energy),
        "steps": int(t + 1),
        "compute_rollouts": int((t + 1) * n),
    }
