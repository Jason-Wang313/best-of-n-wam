from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import numpy as np


class GymRoboticsUnavailableError(ImportError):
    pass


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=float).reshape(-1))
        if parts:
            return np.concatenate(parts)
    return np.asarray(obs, dtype=float).reshape(-1)


def is_gym_robotics_available(env_id: str = "FetchPush-v4") -> tuple[bool, str]:
    if importlib.util.find_spec("gymnasium") is None:
        return False, "gymnasium import not found"
    if importlib.util.find_spec("gymnasium_robotics") is None:
        return False, "gymnasium_robotics import not found"
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401

        env = gym.make(env_id)
        env.reset(seed=0)
        env.close()
        return True, f"{env_id} available"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"{env_id} unavailable: {type(exc).__name__}: {exc}"


@dataclass
class GymRoboticsAdapter:
    """Optional Gymnasium Robotics Fetch adapter.

    Fetch tasks are MuJoCo manipulation environments with dict observations.
    The cloned simulator state includes qpos, qvel, and the sampled goal so
    rollout pools evaluate candidate action sequences from the same task state.
    """

    env_id: str = "FetchPush-v4"
    horizon: int = 12
    success_threshold: float = 0.05
    render_mode: str | None = None

    def __post_init__(self) -> None:
        ok, reason = is_gym_robotics_available(self.env_id)
        if not ok:
            raise GymRoboticsUnavailableError(reason)
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401

        kwargs: dict[str, Any] = {}
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode
        self.env = gym.make(self.env_id, **kwargs)
        obs, _ = self.env.reset(seed=0)
        self.last_obs_raw = obs
        self.last_obs = _flatten_obs(obs)
        self.task_name = self.env_id
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self._nq = int(self.env.unwrapped.data.qpos.size)
        self._nv = int(self.env.unwrapped.data.qvel.size)
        self._goal_dim = int(np.asarray(getattr(self.env.unwrapped, "goal", np.zeros(0)), dtype=float).size)
        self.state_dim = self._nq + self._nv + self._goal_dim

    def close(self) -> None:
        self.env.close()

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.env_id:
            raise ValueError(f"GymRoboticsAdapter only wraps {self.env_id}, got task_id={task_id}")
        obs, _ = self.env.reset(seed=int(seed))
        self.last_obs_raw = obs
        self.last_obs = _flatten_obs(obs)
        return self.get_state()

    def reset_task(self, *, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        data = self.env.unwrapped.data
        goal = np.asarray(getattr(self.env.unwrapped, "goal", np.zeros(0)), dtype=float).reshape(-1)
        return np.concatenate([np.asarray(data.qpos, dtype=float), np.asarray(data.qvel, dtype=float), goal])

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float).reshape(-1)
        expected = self._nq + self._nv + self._goal_dim
        if state.size != expected:
            raise ValueError(f"state has size {state.size}, expected {expected}")
        data = self.env.unwrapped.data
        data.qpos[:] = state[: self._nq]
        data.qvel[:] = state[self._nq : self._nq + self._nv]
        if self._goal_dim:
            self.env.unwrapped.goal = state[self._nq + self._nv :].copy()
        import mujoco

        mujoco.mj_forward(self.env.unwrapped.model, data)
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0
        obs = self.env.unwrapped._get_obs()
        self.last_obs_raw = obs
        self.last_obs = _flatten_obs(obs)

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), self.last_obs])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs_raw = obs
        self.last_obs = _flatten_obs(obs)
        return self.last_obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def achieved_goal(self) -> np.ndarray:
        if isinstance(self.last_obs_raw, dict) and "achieved_goal" in self.last_obs_raw:
            return np.asarray(self.last_obs_raw["achieved_goal"], dtype=float).reshape(-1)
        return self.last_obs[-3:]

    def desired_goal(self) -> np.ndarray:
        if isinstance(self.last_obs_raw, dict) and "desired_goal" in self.last_obs_raw:
            return np.asarray(self.last_obs_raw["desired_goal"], dtype=float).reshape(-1)
        return np.asarray(getattr(self.env.unwrapped, "goal", np.zeros(3)), dtype=float).reshape(-1)

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.achieved_goal() - self.desired_goal()))

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        return self.distance_to_goal() <= self.success_threshold

    def compute_utility(self, state: np.ndarray | None = None) -> float:
        if state is not None:
            self.set_state(state)
        dist = self.distance_to_goal()
        return float(2.5 * (dist <= self.success_threshold) - dist)

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(seed + 9973 * i) for i in range(int(n))]

    def heuristic_action(self, scale: float = 4.0) -> np.ndarray:
        direction = self.desired_goal() - self.achieved_goal()
        action = np.zeros(self.action_dim, dtype=float)
        action[: min(3, self.action_dim, direction.size)] = scale * direction[: min(3, self.action_dim, direction.size)]
        if self.action_dim >= 4:
            action[3] = 0.0
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.gym_robotics_rollouts import sample_rollout_pool

        return sample_rollout_pool(self, *args, **kwargs)

    def score_rollouts(self, rollouts: list[dict[str, Any]], scorer: str = "utility") -> np.ndarray:
        if scorer == "utility":
            return np.asarray([r["utility"] for r in rollouts], dtype=float)
        if scorer == "low_energy":
            return -np.asarray([r["energy"] for r in rollouts], dtype=float)
        if scorer == "random":
            rng = np.random.default_rng(0)
            return rng.normal(size=len(rollouts))
        raise ValueError(f"unknown scorer: {scorer}")

    def evaluate_real_success(self, rollouts: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([float(r["success"]) for r in rollouts], dtype=float)

    def run_closed_loop(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.gym_robotics_rollouts import run_closed_loop

        return run_closed_loop(self, *args, **kwargs)
