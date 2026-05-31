"""Optional LIBERO benchmark integration.

LIBERO is kept optional because the official stack currently expects an older
RoboSuite runtime than the default benchmark environment. The adapter imports
LIBERO lazily and exposes clone-restored state/action rollouts when a compatible
interpreter is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import random
from typing import Any

import numpy as np


class LIBEROUnavailableError(ImportError):
    pass


def is_libero_available() -> tuple[bool, str]:
    if importlib.util.find_spec("libero") is None:
        return False, "libero import not found"
    try:
        from libero.libero import benchmark  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

        return True, "LIBERO imports available"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"LIBERO import failed: {type(exc).__name__}: {exc}"


def _flatten_numeric(values: Any) -> np.ndarray:
    if isinstance(values, dict):
        parts: list[np.ndarray] = []
        for key in sorted(values):
            if "image" in key or "rgb" in key or "depth" in key or "segmentation" in key:
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
class LIBEROAdapter:
    """State/action adapter for small LIBERO rollout-pool experiments.

    Utility is dense and task local: it rewards reducing a distance made from
    end-effector-to-object and object-to-target terms, plus sparse LIBERO
    success when it occurs, minus action energy. This deliberately avoids
    claiming policy-level LIBERO success from tiny random-shooting rollouts.
    """

    suite: str = "libero_spatial"
    task_index: int = 0
    task_order_index: int = 0
    horizon: int = 4
    camera_width: int = 64
    camera_height: int = 64
    controller: str = "OSC_POSE"
    use_camera_obs: bool = False
    has_offscreen_renderer: bool = False
    action_scale: float = 0.65
    gripper_scale: float = 1.0
    target_weight: float = 1.0
    eef_weight: float = 0.5
    success_bonus: float = 5.0
    reward_weight: float = 1.0
    energy_penalty: float = 0.01

    def __post_init__(self) -> None:
        ok, reason = is_libero_available()
        if not ok:
            raise LIBEROUnavailableError(reason)
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self._benchmark_module = benchmark
        self._env_cls = OffScreenRenderEnv
        self._build_env()

    def _build_env(self) -> None:
        bm_cls = self._benchmark_module.get_benchmark(self.suite)
        self.benchmark = bm_cls(task_order_index=int(self.task_order_index))
        if not 0 <= int(self.task_index) < self.benchmark.get_num_tasks():
            raise LIBEROUnavailableError(f"task_index={self.task_index} is outside suite {self.suite}")
        self.task = self.benchmark.get_task(int(self.task_index))
        self.task_name = f"{self.suite}/{self.task.name}"
        bddl_path = self.benchmark.get_task_bddl_file_path(int(self.task_index))
        self.env = self._env_cls(
            bddl_file_name=bddl_path,
            camera_heights=int(self.camera_height),
            camera_widths=int(self.camera_width),
            use_camera_obs=bool(self.use_camera_obs),
            has_offscreen_renderer=bool(self.has_offscreen_renderer),
            horizon=int(self.horizon),
            controller=self.controller,
        )
        self.env.seed(0)
        self.last_obs = self.env.reset()
        low, high = self.env.env.action_spec
        self.action_low = np.asarray(low, dtype=float).reshape(-1)
        self.action_high = np.asarray(high, dtype=float).reshape(-1)
        self.action_dim = int(self.env.env.action_dim)
        self.state_dim = int(self.get_state().size)
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict[str, Any] = {}

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def _reset_to_task(self, task_id: str) -> None:
        if "/" in task_id:
            suite, index_or_name = task_id.split("/", 1)
        else:
            suite, index_or_name = self.suite, task_id
        if index_or_name.isdigit():
            task_index = int(index_or_name)
        else:
            bm_cls = self._benchmark_module.get_benchmark(suite)
            bm = bm_cls(task_order_index=int(self.task_order_index))
            names = bm.get_task_names()
            if index_or_name not in names:
                raise LIBEROUnavailableError(f"unknown LIBERO task {task_id!r}")
            task_index = names.index(index_or_name)
        if suite == self.suite and task_index == self.task_index:
            return
        self.close()
        self.suite = suite
        self.task_index = task_index
        self._build_env()

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None:
            self._reset_to_task(task_id)
        seed = int(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        self.env.seed(seed)
        self.last_obs = self.env.reset()
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        return self.get_state()

    def reset_task(self, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        return np.asarray(self.env.get_sim_state(), dtype=float).reshape(-1)

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float).reshape(-1)
        self.last_obs = self.env.regenerate_obs_from_state(state.copy())
        if hasattr(self.env.env, "timestep"):
            self.env.env.timestep = 0
        if hasattr(self.env.env, "done"):
            self.env.env.done = False
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

    def _raw_obs(self) -> dict[str, Any]:
        try:
            self.last_obs = self.env.env._get_observations()
        except Exception:
            pass
        return self.last_obs if isinstance(self.last_obs, dict) else {}

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), _flatten_numeric(self._raw_obs())])

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        action = rng.uniform(self.action_low, self.action_high)
        scale = np.full_like(action, float(self.action_scale), dtype=float)
        if action.size:
            scale[-1] = float(self.gripper_scale)
        return np.clip(action * scale, self.action_low, self.action_high).astype(float)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, self.action_low, self.action_high)
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = dict(info)
        return _flatten_numeric(obs), self.last_reward, self.last_done, False, self.last_info

    def _obs_vector(self, key: str) -> np.ndarray | None:
        raw = self._raw_obs()
        if key not in raw:
            return None
        try:
            arr = np.asarray(raw[key], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return arr if arr.size else None

    def _position(self, object_name: str) -> np.ndarray | None:
        arr = self._obs_vector(f"{object_name}_pos")
        if arr is not None and arr.size >= 3:
            return arr[:3].copy()
        try:
            body_id = self.env.sim.model.body_name2id(object_name)
            return np.asarray(self.env.sim.data.body_xpos[body_id], dtype=float).reshape(3)
        except Exception:
            return None

    def _eef_pos(self) -> np.ndarray:
        arr = self._obs_vector("robot0_eef_pos")
        if arr is not None and arr.size >= 3:
            return arr[:3].copy()
        try:
            site_id = self.env.robots[0].eef_site_id
            if isinstance(site_id, dict):
                site_id = next(iter(site_id.values()))
            return np.asarray(self.env.sim.data.site_xpos[int(site_id)], dtype=float).reshape(3)
        except Exception:
            return np.zeros(3, dtype=float)

    def task_distance(self) -> float:
        objects = list(getattr(self.env, "obj_of_interest", []) or [])
        if not objects:
            return 0.0
        obj = self._position(objects[0])
        eef = self._eef_pos()
        if obj is None:
            return float(np.linalg.norm(eef))
        distance = self.eef_weight * float(np.linalg.norm(obj - eef))
        if len(objects) >= 2:
            target = self._position(objects[1])
            if target is not None:
                distance += self.target_weight * float(np.linalg.norm(obj - target))
        return float(distance)

    def object_distance(self) -> float:
        return self.task_distance()

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        try:
            return bool(self.env.check_success())
        except Exception:
            return bool((getattr(self, "last_info", {}) or {}).get("success", False))

    def evaluate_real_success(self, state: np.ndarray | None = None) -> bool:
        return self.evaluate_success(state)

    def compute_utility(
        self,
        initial_distance: float | None = None,
        energy: float = 0.0,
        total_reward: float = 0.0,
        state: np.ndarray | None = None,
    ) -> float:
        if state is not None:
            self.set_state(state)
        final_distance = self.task_distance()
        progress = 0.0 if initial_distance is None else float(initial_distance - final_distance)
        return float(
            progress
            + self.reward_weight * float(total_reward)
            + self.success_bonus * float(self.evaluate_success())
            - self.energy_penalty * float(energy)
        )

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
        initial_distance = self.task_distance()
        for rollout_id in range(int(n_rollouts)):
            self.set_state(initial_state)
            total_reward = 0.0
            terminal_reward = 0.0
            energy = 0.0
            steps = 0
            for t in range(horizon):
                flat_action = self.sample_action(rng)
                actions[rollout_id, t] = flat_action
                energy += float(np.sum(flat_action * flat_action))
                _, reward, terminated, truncated, _ = self.step(flat_action)
                total_reward += float(reward)
                terminal_reward = float(reward)
                steps += 1
                if terminated or truncated:
                    break
            final_distance = self.task_distance()
            success = self.evaluate_success()
            progress = float(initial_distance - final_distance)
            utility = self.compute_utility(initial_distance, energy, total_reward)
            records.append(
                {
                    "rollout_id": int(rollout_id),
                    "initial_distance": float(initial_distance),
                    "final_distance": float(final_distance),
                    "progress": progress,
                    "energy": float(energy),
                    "steps": int(steps),
                    "total_reward": float(total_reward),
                    "terminal_reward": float(terminal_reward),
                    "success": bool(success),
                    "utility": float(utility),
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
        reward = np.asarray([r["total_reward"] for r in records], dtype=float)
        rng = np.random.default_rng(int(seed))
        return {
            "random": rng.normal(size=len(records)),
            "distance_progress": progress - final_distance,
            "low_energy": -energy,
            "benchmark_reward": reward,
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
        start_distance = self.task_distance()
        total_energy = 0.0
        total_reward = 0.0
        for _ in range(int(steps)):
            state = self.get_state()
            pool = self.sample_rollouts(
                state,
                n_rollouts=int(n),
                horizon=candidate_horizon or self.horizon,
                seed=int(rng.integers(1_000_000_000)),
            )
            scores = self.score_rollouts(pool, seed=int(rng.integers(1_000_000_000))).get(scorer)
            if scores is None:
                raise ValueError(f"unknown scorer {scorer!r}")
            best = int(np.argmax(scores))
            action = pool["actions"][best, 0]
            total_energy += float(np.sum(action * action))
            _, reward, terminated, truncated, _ = self.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        final_distance = self.task_distance()
        success = self.evaluate_success()
        return {
            "start_distance": float(start_distance),
            "final_distance": float(final_distance),
            "progress": float(start_distance - final_distance),
            "energy": float(total_energy),
            "total_reward": float(total_reward),
            "success": bool(success),
            "utility": float(
                start_distance
                - final_distance
                + self.reward_weight * total_reward
                + self.success_bonus * float(success)
                - self.energy_penalty * total_energy
            ),
        }
