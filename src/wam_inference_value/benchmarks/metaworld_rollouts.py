from __future__ import annotations

from typing import Any

import numpy as np

from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature


def sample_action_sequences(adapter: Any, n_rollouts: int, horizon: int, seed: int, state: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray(adapter.env.action_space.low, dtype=float).reshape(-1)
    high = np.asarray(adapter.env.action_space.high, dtype=float).reshape(-1)
    actions = rng.normal(0.0, 0.55, size=(int(n_rollouts), int(horizon), adapter.action_dim))
    actions = np.clip(actions, low, high)

    n_guided = min(int(n_rollouts), max(1, int(0.35 * int(n_rollouts))))
    for i in range(n_guided):
        if state is not None:
            adapter.set_state(state)
        guided = []
        for _ in range(int(horizon)):
            base = adapter.heuristic_action(scale=float(rng.uniform(3.0, 7.0)))
            guided.append(np.clip(base + rng.normal(0.0, 0.15, size=adapter.action_dim), low, high))
            adapter.step(guided[-1])
        actions[i] = np.asarray(guided, dtype=float)
    rng.shuffle(actions, axis=0)
    return actions


def rollout_sequence(adapter: Any, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    adapter.set_state(state)
    _, initial_info = adapter.current_eval()
    initial_distance = float(initial_info.get("obj_to_target", adapter.distance_to_goal()))
    energy = 0.0
    total_reward = 0.0
    final_info: dict[str, Any] = dict(initial_info)
    for action in np.asarray(actions, dtype=float):
        energy += float(np.sum(np.asarray(action, dtype=float) ** 2))
        _, reward, terminated, truncated, info = adapter.step(action)
        total_reward += float(reward)
        final_info = dict(info)
        if terminated or truncated:
            break
    final_state = adapter.get_state()
    final_distance = float(final_info.get("obj_to_target", adapter.distance_to_goal()))
    success = bool(float(final_info.get("success", 0.0)) >= 0.5)
    progress = initial_distance - final_distance
    terminal_reward = float(final_info.get("unscaled_reward", total_reward))
    utility = terminal_reward + 4.0 * float(success) + 3.0 * progress - 0.02 * energy
    return {
        "initial_distance": float(initial_distance),
        "final_distance": float(final_distance),
        "progress": float(progress),
        "energy": float(energy),
        "success": bool(success),
        "utility": float(utility),
        "total_reward": float(total_reward),
        "terminal_reward": float(terminal_reward),
        "final_state": final_state,
        "final_info": final_info,
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
    steps: int = 8,
    candidate_horizon: int = 6,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = adapter.reset(seed)
    _, initial_info = adapter.current_eval()
    initial_distance = float(initial_info.get("obj_to_target", adapter.distance_to_goal()))
    energy = 0.0
    total_reward = 0.0
    t = 0
    for t in range(int(steps)):
        actions = sample_action_sequences(adapter, int(n), int(candidate_horizon), seed + 7919 * (t + 1), state)
        if scorer == "random":
            scores = rng.normal(size=int(n))
        elif scorer == "oracle":
            scores = np.asarray([rollout_sequence(adapter, state, seq)["utility"] for seq in actions], dtype=float)
        elif scorer == "reward":
            scores = np.asarray([rollout_sequence(adapter, state, seq)["total_reward"] for seq in actions], dtype=float)
        elif scorer == "learned":
            x = benchmark_feature(adapter.feature_state(state), actions, int(feature_horizon or candidate_horizon))
            scores = np.asarray(model.predict(x)[:, 0], dtype=float)
        else:
            raise ValueError(f"unknown closed-loop scorer: {scorer}")
        best = int(np.argmax(scores))
        first_action = actions[best, 0]
        energy += float(np.sum(first_action * first_action))
        adapter.set_state(state)
        _, reward, terminated, truncated, info = adapter.step(first_action)
        total_reward += float(reward)
        state = adapter.get_state()
        if bool(float(info.get("success", 0.0)) >= 0.5) or terminated or truncated:
            break
    _, final_info = adapter.current_eval()
    final_distance = float(final_info.get("obj_to_target", adapter.distance_to_goal()))
    success = bool(float(final_info.get("success", 0.0)) >= 0.5)
    utility = float(final_info.get("unscaled_reward", total_reward)) + 4.0 * float(success) + 3.0 * (initial_distance - final_distance) - 0.02 * energy
    return {
        "success": float(success),
        "utility": float(utility),
        "final_distance": float(final_distance),
        "energy": float(energy),
        "total_reward": float(total_reward),
        "steps": int(t + 1),
        "compute_rollouts": int((t + 1) * n),
    }
