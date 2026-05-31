"""Optional Meta-World benchmark integration."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import numpy as np


class MetaWorldUnavailableError(ImportError):
    pass


def is_metaworld_available(task_name: str = "reach-v3") -> tuple[bool, str]:
    if importlib.util.find_spec("metaworld") is None:
        return False, "metaworld import not found"
    try:
        import metaworld

        if task_name not in metaworld.ALL_V3_ENVIRONMENTS:
            return False, f"{task_name} not in Meta-World v3 registry"
        ml1 = metaworld.ML1(task_name)
        env = ml1.train_classes[task_name]()
        env.set_task(ml1.train_tasks[0])
        env.reset(seed=0)
        env.close()
        return True, f"{task_name} available"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"{task_name} unavailable: {type(exc).__name__}: {exc}"


@dataclass
class MetaWorldAdapter:
    """Small state adapter for Meta-World ML1 tasks.

    Meta-World exposes qpos/qvel cloning but task goals live in environment
    attributes, so the adapter stores qpos, qvel, and the current target
    together. The benchmark remains optional; importing this module alone does
    not import Meta-World.
    """

    task_name: str = "reach-v3"
    horizon: int = 8
    task_index: int = 0
    success_bonus: float = 5.0
    energy_penalty: float = 0.003

    def __post_init__(self) -> None:
        ok, reason = is_metaworld_available(self.task_name)
        if not ok:
            raise MetaWorldUnavailableError(reason)
        import metaworld

        self._benchmark = metaworld.ML1(self.task_name)
        self._tasks = list(self._benchmark.train_tasks)
        self.env = self._benchmark.train_classes[self.task_name]()
        self.env.set_task(self._tasks[int(self.task_index) % len(self._tasks)])
        obs, _ = self.env.reset(seed=0)
        self.last_obs = np.asarray(obs, dtype=float).reshape(-1)
        self.last_info: dict[str, Any] = {}
        self.last_reward = 0.0
        self.action_dim = int(np.prod(self.env.action_space.shape))
        qpos, qvel = self.env.get_env_state()
        self._nq = int(np.asarray(qpos).size)
        self._nv = int(np.asarray(qvel).size)
        self._goal_dim = int(self._goal().size)
        self.state_dim = self._nq + self._nv + self._goal_dim

    def close(self) -> None:
        self.env.close()

    def _set_task_for_seed(self, seed: int) -> None:
        task = self._tasks[int(seed) % len(self._tasks)]
        self.env.set_task(task)

    def _goal(self) -> np.ndarray:
        if hasattr(self.env, "_target_pos"):
            return np.asarray(getattr(self.env, "_target_pos"), dtype=float).reshape(-1)
        if hasattr(self.env, "goal"):
            return np.asarray(getattr(self.env, "goal"), dtype=float).reshape(-1)
        return np.zeros(3, dtype=float)

    def _set_goal(self, goal: np.ndarray) -> None:
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if hasattr(self.env, "_target_pos"):
            setattr(self.env, "_target_pos", goal.copy())
        try:
            setattr(self.env, "goal", goal.copy())
        except Exception:
            pass
        try:
            self.env.model.site("goal").pos = goal.copy()
        except Exception:
            pass

    def _forward(self) -> None:
        try:
            import mujoco

            mujoco.mj_forward(self.env.model, self.env.data)
        except Exception:
            pass

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.task_name:
            self.close()
            self.task_name = task_id
            self.__post_init__()
        self._set_task_for_seed(seed)
        obs, _ = self.env.reset(seed=int(seed))
        self.last_obs = np.asarray(obs, dtype=float).reshape(-1)
        self.last_info = {}
        self.last_reward = 0.0
        return self.get_state()

    def reset_task(self, *, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        qpos, qvel = self.env.get_env_state()
        return np.concatenate(
            [
                np.asarray(qpos, dtype=float).reshape(-1),
                np.asarray(qvel, dtype=float).reshape(-1),
                self._goal(),
            ]
        )

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float).reshape(-1)
        expected = self._nq + self._nv + self._goal_dim
        if state.size != expected:
            raise ValueError(f"state has size {state.size}, expected {expected}")
        qpos = state[: self._nq].copy()
        qvel = state[self._nq : self._nq + self._nv].copy()
        goal = state[self._nq + self._nv :].copy()
        self._set_goal(goal)
        self.env.set_env_state((qpos, qvel))
        if hasattr(self.env, "curr_path_length"):
            self.env.curr_path_length = 0
        self._forward()
        self.last_obs = np.asarray(self.env._get_obs(), dtype=float).reshape(-1)

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), self.last_obs])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = np.asarray(obs, dtype=float).reshape(-1)
        self.last_reward = float(reward)
        self.last_info = dict(info)
        return self.last_obs, self.last_reward, bool(terminated), bool(truncated), self.last_info

    def current_eval(self) -> tuple[float, dict[str, Any]]:
        obs = np.asarray(self.env._get_obs(), dtype=float).reshape(-1)
        action = np.zeros(self.action_dim, dtype=np.float32)
        try:
            reward, info = self.env.evaluate_state(obs, action)
            return float(reward), dict(info)
        except Exception:
            dist = self.distance_to_goal_from_obs(obs)
            return float(-dist), {"success": float(dist <= 0.05), "obj_to_target": float(dist), "unscaled_reward": float(-dist)}

    def distance_to_goal_from_obs(self, obs: np.ndarray | None = None) -> float:
        obs = self.last_obs if obs is None else np.asarray(obs, dtype=float).reshape(-1)
        goal = self._goal()
        if obs.size >= 7 and goal.size >= 3 and "reach" not in self.task_name:
            pos = obs[4:7]
        else:
            pos = obs[:3]
        return float(np.linalg.norm(pos[: min(3, goal.size)] - goal[: min(3, pos.size)]))

    def distance_to_goal(self) -> float:
        _, info = self.current_eval()
        if "obj_to_target" in info:
            return float(info["obj_to_target"])
        return self.distance_to_goal_from_obs()

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        _, info = self.current_eval()
        return bool(float(info.get("success", 0.0)) >= 0.5)

    def compute_utility(self, state: np.ndarray | None = None) -> float:
        if state is not None:
            self.set_state(state)
        reward, info = self.current_eval()
        dist = float(info.get("obj_to_target", self.distance_to_goal_from_obs()))
        success = float(info.get("success", 0.0) >= 0.5)
        return float(reward + self.success_bonus * success - 0.25 * dist)

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(seed + 9973 * i) for i in range(int(n))]

    def heuristic_action(self, scale: float = 5.0) -> np.ndarray:
        obs = np.asarray(self.last_obs, dtype=float).reshape(-1)
        tcp = obs[:3]
        obj = obs[4:7] if obs.size >= 7 else tcp
        goal = self._goal()[:3]
        if "reach" in self.task_name:
            direction = goal - tcp
            grip = 0.0
        else:
            tcp_to_obj = float(np.linalg.norm(tcp - obj))
            if tcp_to_obj > 0.04:
                direction = obj - tcp
                grip = 0.6
            else:
                direction = goal - obj
                grip = -0.8 if "pick" in self.task_name else 0.2
        action = np.zeros(self.action_dim, dtype=float)
        action[: min(3, self.action_dim)] = scale * direction[: min(3, self.action_dim)]
        if self.action_dim >= 4:
            action[3] = grip
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.metaworld_rollouts import sample_rollout_pool

        return sample_rollout_pool(self, *args, **kwargs)

    def score_rollouts(self, rollouts: list[dict[str, Any]], scorer: str = "utility") -> np.ndarray:
        if scorer == "utility":
            return np.asarray([r["utility"] for r in rollouts], dtype=float)
        if scorer == "reward":
            return np.asarray([r["total_reward"] for r in rollouts], dtype=float)
        if scorer == "low_energy":
            return -np.asarray([r["energy"] for r in rollouts], dtype=float)
        if scorer == "random":
            return np.random.default_rng(0).normal(size=len(rollouts))
        raise ValueError(f"unknown scorer: {scorer}")

    def evaluate_real_success(self, rollouts: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([float(r["success"]) for r in rollouts], dtype=float)

    def run_closed_loop(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.metaworld_rollouts import run_closed_loop

        return run_closed_loop(self, *args, **kwargs)
