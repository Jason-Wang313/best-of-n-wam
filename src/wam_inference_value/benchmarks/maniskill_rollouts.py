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
    adapter._ensure_env()
    rng = np.random.default_rng(seed)
    low = np.asarray(adapter.env.action_space.low, dtype=float).reshape(-1)
    high = np.asarray(adapter.env.action_space.high, dtype=float).reshape(-1)
    dim = int(adapter.action_dim)
    actions = np.zeros((int(n_rollouts), int(horizon), dim), dtype=float)
    for i in range(int(n_rollouts)):
        mode = rng.choice(["small", "medium", "burst", "repeat"], p=[0.30, 0.38, 0.20, 0.12])
        if mode == "small":
            scale = 0.18
        elif mode == "medium":
            scale = 0.42
        elif mode == "burst":
            scale = 0.78
        else:
            scale = 0.55
        base = rng.normal(0.0, scale, size=(dim,))
        for t in range(int(horizon)):
            if mode == "repeat":
                action = base + rng.normal(0.0, 0.08, size=(dim,))
            else:
                decay = 1.0 - 0.35 * (t / max(1, int(horizon) - 1))
                action = decay * rng.normal(0.0, scale, size=(dim,))
            if dim >= 2:
                # The final gripper command often creates useful variation.
                action[-1] = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0)
            actions[i, t] = np.clip(action, low, high)
    return actions


def rollout_sequence(adapter: Any, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    adapter.set_state(state)
    initial_state = adapter.get_state()
    energy = 0.0
    total_reward = 0.0
    terminated = False
    truncated = False
    for action in np.asarray(actions, dtype=float):
        energy += float(np.sum(action * action))
        _, reward, terminated, truncated, _ = adapter.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    final_state = adapter.get_state()
    success = adapter.evaluate_success()
    state_delta = float(np.linalg.norm(final_state - initial_state))
    utility = total_reward + adapter.success_bonus * float(success) + 0.02 * state_delta - adapter.energy_penalty * energy
    return {
        "energy": float(energy),
        "success": bool(success),
        "utility": float(utility),
        "total_reward": float(total_reward),
        "state_delta": state_delta,
        "final_state": final_state,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
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
    steps: int = 6,
    candidate_horizon: int = 6,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = adapter.reset(seed)
    energy = 0.0
    total_reward = 0.0
    for t in range(int(steps)):
        actions = sample_action_sequences(adapter, int(n), int(candidate_horizon), seed + 7919 * (t + 1))
        feature_state = adapter.feature_state(state)
        if scorer == "random":
            scores = rng.normal(size=int(n))
        elif scorer == "oracle":
            scores = np.asarray([rollout_sequence(adapter, state, seq)["utility"] for seq in actions], dtype=float)
        elif scorer == "learned":
            x = benchmark_feature(feature_state, actions, int(feature_horizon or candidate_horizon))
            scores = np.asarray(model.predict(x)[:, -1], dtype=float)
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
    success = adapter.evaluate_success()
    utility = total_reward + adapter.success_bonus * float(success) - adapter.energy_penalty * energy
    return {
        "success": float(success),
        "utility": float(utility),
        "energy": float(energy),
        "steps": int(t + 1),
        "compute_rollouts": int((t + 1) * n),
    }
