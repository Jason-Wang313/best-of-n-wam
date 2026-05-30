from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import numpy as np


class GymManipUnavailableError(ImportError):
    pass


def is_gym_manip_available(env_id: str = "Reacher-v5") -> tuple[bool, str]:
    if importlib.util.find_spec("gymnasium") is None:
        return False, "gymnasium import not found"
    try:
        import gymnasium as gym

        env = gym.make(env_id)
        env.reset(seed=0)
        env.close()
        return True, f"{env_id} available"
    except Exception as exc:  # pragma: no cover - depends on optional local deps
        return False, f"{env_id} unavailable: {type(exc).__name__}: {exc}"


@dataclass
class GymManipAdapter:
    """State-based Gymnasium/MuJoCo manipulation fallback adapter.

    The default task is Reacher-v5 because it is small, deterministic under
    seed, CPU-friendly, and exposes qpos/qvel state cloning through MuJoCo.
    """

    env_id: str = "Reacher-v5"
    horizon: int = 15
    success_threshold: float = 0.07
    render_mode: str | None = None

    def __post_init__(self) -> None:
        ok, reason = is_gym_manip_available(self.env_id)
        if not ok:
            raise GymManipUnavailableError(reason)
        import gymnasium as gym

        kwargs: dict[str, Any] = {}
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode
        self.env = gym.make(self.env_id, **kwargs)
        obs, _ = self.env.reset(seed=0)
        self.last_obs = np.asarray(obs, dtype=float)
        self.task_name = self.env_id
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.state_dim = int(self.env.unwrapped.data.qpos.size + self.env.unwrapped.data.qvel.size)

    def close(self) -> None:
        self.env.close()

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.env_id:
            raise ValueError(f"GymManipAdapter only wraps {self.env_id}, got task_id={task_id}")
        obs, _ = self.env.reset(seed=int(seed))
        self.last_obs = np.asarray(obs, dtype=float)
        return self.get_state()

    def reset_task(self, *, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        data = self.env.unwrapped.data
        return np.concatenate([np.asarray(data.qpos, dtype=float), np.asarray(data.qvel, dtype=float)])

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float)
        nq = int(self.env.unwrapped.data.qpos.size)
        nv = int(self.env.unwrapped.data.qvel.size)
        if state.size != nq + nv:
            raise ValueError(f"state has size {state.size}, expected {nq + nv}")
        self.env.unwrapped.set_state(state[:nq], state[nq:])
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0
        self.last_obs = np.asarray(self.env.unwrapped._get_obs(), dtype=float)

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), np.asarray(self.last_obs, dtype=float)])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        obs, reward, terminated, truncated, info = self.env.step(np.clip(action, self.env.action_space.low, self.env.action_space.high))
        self.last_obs = np.asarray(obs, dtype=float)
        return self.last_obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def distance_to_goal(self) -> float:
        # Reacher-v5 exposes the fingertip-target vector in the last two obs entries.
        return float(np.linalg.norm(np.asarray(self.last_obs[-2:], dtype=float)))

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        return self.distance_to_goal() <= self.success_threshold

    def compute_utility(self, state: np.ndarray | None = None) -> float:
        if state is not None:
            self.set_state(state)
        dist = self.distance_to_goal()
        return float(2.0 * (dist <= self.success_threshold) - dist)

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(seed + 9973 * i) for i in range(int(n))]

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.gym_manip_rollouts import sample_rollout_pool

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
        from wam_inference_value.benchmarks.gym_manip_rollouts import run_closed_loop

        return run_closed_loop(self, *args, **kwargs)
