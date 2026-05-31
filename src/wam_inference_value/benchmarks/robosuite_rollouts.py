from __future__ import annotations

from typing import Any

import numpy as np

from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature


def sample_action_sequences(
    adapter: Any,
    n_rollouts: int,
    horizon: int,
    seed: int,
    state: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray(adapter.action_low, dtype=float).reshape(-1)
    high = np.asarray(adapter.action_high, dtype=float).reshape(-1)
    n_rollouts = int(n_rollouts)
    horizon = int(horizon)
    actions = rng.normal(0.0, 0.48, size=(n_rollouts, horizon, adapter.action_dim))
    burst = rng.random(size=(n_rollouts, 1, 1)) < 0.20
    actions = np.where(burst, actions * 1.7, actions)
    actions = np.clip(actions, low, high)

    n_guided = min(n_rollouts, max(1, int(0.45 * n_rollouts)))
    for i in range(n_guided):
        if state is not None:
            adapter.set_state(state)
        guided = []
        for step in range(horizon):
            scale = float(rng.uniform(4.0, 8.0))
            base = adapter.heuristic_action(scale=scale)
            if step > max(1, horizon // 2):
                base[: min(3, adapter.action_dim)] += rng.normal(0.0, 0.18, size=min(3, adapter.action_dim))
            guided_action = np.clip(base + rng.normal(0.0, 0.10, size=adapter.action_dim), low, high)
            guided.append(guided_action)
            try:
                _, _, terminated, truncated, _ = adapter.step(guided_action)
            except ValueError:
                break
            if terminated or truncated:
                break
        while len(guided) < horizon:
            guided.append(np.zeros(adapter.action_dim, dtype=float))
        actions[i] = np.asarray(guided, dtype=float)
    rng.shuffle(actions, axis=0)
    return actions


def rollout_sequence(adapter: Any, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    adapter.set_state(state)
    initial_distance = adapter.distance_to_goal()
    initial_progress = adapter.task_progress()
    initial_reward = adapter.current_reward()
    energy = 0.0
    total_reward = 0.0
    terminal_reward = initial_reward
    for action in np.asarray(actions, dtype=float):
        energy += float(np.sum(np.asarray(action, dtype=float) ** 2))
        _, reward, terminated, truncated, _ = adapter.step(action)
        total_reward += float(reward)
        terminal_reward = float(reward)
        if terminated or truncated:
            break
    final_state = adapter.get_state()
    final_distance = adapter.distance_to_goal()
    final_progress = adapter.task_progress()
    success = adapter.evaluate_success()
    progress = initial_distance - final_distance
    dense_progress = final_progress - initial_progress
    utility = (
        2.5 * total_reward
        + terminal_reward
        + 4.0 * float(success)
        + 2.0 * progress
        + 2.0 * dense_progress
        - adapter.energy_penalty * energy
    )
    return {
        "initial_distance": float(initial_distance),
        "final_distance": float(final_distance),
        "progress": float(progress),
        "dense_progress": float(dense_progress),
        "energy": float(energy),
        "success": bool(success),
        "utility": float(utility),
        "total_reward": float(total_reward),
        "terminal_reward": float(terminal_reward),
        "initial_reward": float(initial_reward),
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
    steps: int = 6,
    candidate_horizon: int = 6,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = adapter.reset(seed)
    initial_distance = adapter.distance_to_goal()
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
        _, reward, terminated, truncated, _ = adapter.step(first_action)
        total_reward += float(reward)
        state = adapter.get_state()
        if adapter.evaluate_success() or terminated or truncated:
            break
    final_distance = adapter.distance_to_goal()
    success = adapter.evaluate_success()
    utility = 2.5 * total_reward + 4.0 * float(success) + 2.0 * (initial_distance - final_distance) - adapter.energy_penalty * energy
    return {
        "success": float(success),
        "utility": float(utility),
        "final_distance": float(final_distance),
        "energy": float(energy),
        "total_reward": float(total_reward),
        "steps": int(t + 1),
        "compute_rollouts": int((t + 1) * n),
    }
