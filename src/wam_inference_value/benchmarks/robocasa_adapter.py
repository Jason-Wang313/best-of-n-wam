"""Optional RoboCasa benchmark integration.

RoboCasa is intentionally optional and is heavier than the canonical smoke
suite: current RoboCasa365 wheels pin MuJoCo 3.3.1 and require a separate
kitchen asset download. This adapter imports RoboCasa lazily and gives scripts
a narrow, state-restorable interface when those dependencies are present.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import random
from typing import Any

import numpy as np


class RoboCasaUnavailableError(ImportError):
    pass


ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


def is_robocasa_available() -> tuple[bool, str]:
    if importlib.util.find_spec("robocasa") is None:
        return False, "robocasa import not found"
    if importlib.util.find_spec("gymnasium") is None:
        return False, "gymnasium import not found"
    try:
        import robocasa  # noqa: F401
        import mujoco
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"robocasa import failed: {type(exc).__name__}: {exc}"
    return True, f"robocasa import available with mujoco {getattr(mujoco, '__version__', 'unknown')}"


def _flatten_numeric(values: Any) -> np.ndarray:
    if isinstance(values, dict):
        parts: list[np.ndarray] = []
        for key in sorted(values):
            if "image" in key or "language" in key or "annotation" in key:
                continue
            try:
                arr = np.asarray(values[key], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size:
                parts.append(arr)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


@dataclass
class RoboCasaAdapter:
    """Small state/action adapter for RoboCasa kitchen tasks.

    The adapter exposes clone/set-state rollouts through MuJoCo state vectors.
    It computes a dense utility from end-effector progress toward the target
    object plus sparse task success, so a tiny random-shooting probe can produce
    non-degenerate utility curves even when random actions do not solve the
    full kitchen task.
    """

    env_id: str = "robocasa/PickPlaceCounterToCabinet"
    split: str = "pretrain"
    horizon: int = 3
    camera_width: int = 16
    camera_height: int = 16
    success_bonus: float = 5.0
    energy_penalty: float = 0.01
    freeze_base: bool = True

    def __post_init__(self) -> None:
        ok, reason = is_robocasa_available()
        if not ok:
            raise RoboCasaUnavailableError(reason)
        import gymnasium as gym
        import robocasa  # noqa: F401 - registers gymnasium environments

        self.task_name = self.env_id
        self.env = gym.make(
            self.env_id,
            split=self.split,
            seed=0,
            camera_widths=int(self.camera_width),
            camera_heights=int(self.camera_height),
        )
        self.inner = self.env.unwrapped.env
        self.last_obs, self.last_info = self.env.reset(seed=0)
        self._action_slices = self._build_action_slices()
        self.action_low = np.concatenate([self._space_bounds(k)[0] for k in ACTION_KEYS])
        self.action_high = np.concatenate([self._space_bounds(k)[1] for k in ACTION_KEYS])
        self.action_dim = int(self.action_low.size)
        self.state_dim = int(self.get_state().size)
        self.last_reward = 0.0
        self.last_done = False

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def _build_action_slices(self) -> dict[str, slice]:
        start = 0
        out: dict[str, slice] = {}
        for key in ACTION_KEYS:
            space = self.env.action_space[key]
            dim = int(np.prod(space.shape))
            out[key] = slice(start, start + dim)
            start += dim
        return out

    def _space_bounds(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        space = self.env.action_space[key]
        low = np.asarray(space.low, dtype=float).reshape(-1)
        high = np.asarray(space.high, dtype=float).reshape(-1)
        return low, high

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.env_id:
            self.close()
            self.env_id = task_id
            self.__post_init__()
        seed = int(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        self.last_obs, self.last_info = self.env.reset(seed=seed)
        self.last_reward = 0.0
        self.last_done = False
        return self.get_state()

    def reset_task(self, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        return np.asarray(self.inner.sim.get_state().flatten(), dtype=float).reshape(-1)

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float).reshape(-1)
        self.inner.sim.set_state_from_flattened(state.copy())
        self.inner.sim.forward()
        if hasattr(self.inner, "timestep"):
            self.inner.timestep = 0
        if hasattr(self.inner, "done"):
            self.inner.done = False
        self.last_obs = self._raw_obs()
        self.last_reward = 0.0
        self.last_done = False

    def _raw_obs(self) -> dict[str, Any]:
        return self.inner._get_observations(force_update=True)

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), _flatten_numeric(self._raw_obs())])

    def flatten_action(self, action: dict[str, Any]) -> np.ndarray:
        return np.concatenate([np.asarray(action[k], dtype=float).reshape(-1) for k in ACTION_KEYS])

    def unflatten_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"action has size {action.size}, expected {self.action_dim}")
        action = np.clip(action, self.action_low, self.action_high)
        out: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            space = self.env.action_space[key]
            arr = action[self._action_slices[key]].reshape(space.shape).astype(np.float32)
            out[key] = arr
        if self.freeze_base:
            out["action.base_motion"] = np.zeros_like(out["action.base_motion"])
        return out

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        action = rng.uniform(self.action_low, self.action_high)
        if self.freeze_base:
            slc = self._action_slices["action.base_motion"]
            action[slc] = 0.0
        return action.astype(float)

    def step(self, action: np.ndarray | dict[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_dict = self.unflatten_action(action) if not isinstance(action, dict) else action
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        self.last_obs = obs
        self.last_reward = float(reward)
        self.last_done = bool(terminated or truncated)
        self.last_info = dict(info)
        return _flatten_numeric(obs), self.last_reward, bool(terminated), bool(truncated), self.last_info

    def object_distance(self) -> float:
        raw = self._raw_obs()
        if "obj_to_robot0_eef_pos" in raw:
            return float(np.linalg.norm(np.asarray(raw["obj_to_robot0_eef_pos"], dtype=float).reshape(-1)[:3]))
        if "obj_pos" in raw and "robot0_eef_pos" in raw:
            obj = np.asarray(raw["obj_pos"], dtype=float).reshape(-1)[:3]
            eef = np.asarray(raw["robot0_eef_pos"], dtype=float).reshape(-1)[:3]
            return float(np.linalg.norm(obj - eef))
        return 0.0

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        try:
            return bool(self.inner._check_success())
        except Exception:
            return bool((getattr(self, "last_info", {}) or {}).get("success", False))

    def evaluate_real_success(self, state: np.ndarray | None = None) -> bool:
        return self.evaluate_success(state)

    def compute_utility(
        self,
        initial_distance: float | None = None,
        energy: float = 0.0,
        state: np.ndarray | None = None,
    ) -> float:
        if state is not None:
            self.set_state(state)
        final_distance = self.object_distance()
        progress = 0.0 if initial_distance is None else float(initial_distance - final_distance)
        return float(progress + self.success_bonus * float(self.evaluate_success()) - self.energy_penalty * float(energy))

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(int(seed) + 1009 * i) for i in range(int(n))]

    def sample_rollouts(
        self,
        initial_state: np.ndarray | None = None,
        n_rollouts: int = 16,
        horizon: int | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        if initial_state is None:
            initial_state = self.get_state()
        horizon = int(self.horizon if horizon is None else horizon)
        rng = np.random.default_rng(int(seed))
        records: list[dict[str, Any]] = []
        actions = np.zeros((int(n_rollouts), horizon, self.action_dim), dtype=float)
        self.set_state(initial_state)
        initial_distance = self.object_distance()
        for rollout_id in range(int(n_rollouts)):
            self.set_state(initial_state)
            total_reward = 0.0
            terminal_reward = 0.0
            energy = 0.0
            for t in range(horizon):
                flat_action = self.sample_action(rng)
                actions[rollout_id, t] = flat_action
                action_dict = self.unflatten_action(flat_action)
                energy += float(sum(np.sum(np.square(v)) for v in action_dict.values()))
                _, reward, terminated, truncated, info = self.step(action_dict)
                total_reward += float(reward)
                terminal_reward = float(reward)
                if terminated or truncated:
                    break
            final_distance = self.object_distance()
            success = self.evaluate_success()
            progress = float(initial_distance - final_distance)
            utility = float(progress + self.success_bonus * float(success) - self.energy_penalty * energy)
            records.append(
                {
                    "rollout_id": int(rollout_id),
                    "initial_distance": float(initial_distance),
                    "final_distance": float(final_distance),
                    "progress": progress,
                    "energy": float(energy),
                    "total_reward": float(total_reward),
                    "terminal_reward": float(terminal_reward),
                    "success": bool(success),
                    "utility": utility,
                }
            )
        self.set_state(initial_state)
        return {"actions": actions, "records": records}

    def score_rollouts(self, pool: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
        records = pool["records"]
        utility = np.asarray([r["utility"] for r in records], dtype=float)
        progress = np.asarray([r["progress"] for r in records], dtype=float)
        final_distance = np.asarray([r["final_distance"] for r in records], dtype=float)
        energy = np.asarray([r["energy"] for r in records], dtype=float)
        rng = np.random.default_rng(int(seed))
        return {
            "random": rng.normal(size=len(records)),
            "distance_progress": progress - final_distance,
            "low_energy": -energy,
            "oracle_real_utility": utility,
        }

    def run_closed_loop(
        self,
        n: int = 4,
        steps: int = 2,
        candidate_horizon: int | None = None,
        scorer: str = "distance_progress",
        seed: int = 0,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        start_distance = self.object_distance()
        total_energy = 0.0
        total_reward = 0.0
        for step_id in range(int(steps)):
            state = self.get_state()
            pool = self.sample_rollouts(state, n_rollouts=int(n), horizon=candidate_horizon or self.horizon, seed=int(rng.integers(1_000_000_000)))
            scores = self.score_rollouts(pool, seed=int(rng.integers(1_000_000_000))).get(scorer)
            if scores is None:
                raise ValueError(f"unknown scorer {scorer!r}")
            best = int(np.argmax(scores))
            action = pool["actions"][best, 0]
            action_dict = self.unflatten_action(action)
            total_energy += float(sum(np.sum(np.square(v)) for v in action_dict.values()))
            _, reward, terminated, truncated, _ = self.step(action_dict)
            total_reward += float(reward)
            if terminated or truncated:
                break
        final_distance = self.object_distance()
        return {
            "start_distance": float(start_distance),
            "final_distance": float(final_distance),
            "progress": float(start_distance - final_distance),
            "energy": float(total_energy),
            "total_reward": float(total_reward),
            "success": bool(self.evaluate_success()),
            "utility": float(start_distance - final_distance + self.success_bonus * float(self.evaluate_success()) - self.energy_penalty * total_energy),
        }
