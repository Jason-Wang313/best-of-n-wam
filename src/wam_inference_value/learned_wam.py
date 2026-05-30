"""CPU-only learned WAM-lite dynamics for BlockPush2D.

The model is intentionally small: a deterministic feature map over
state/action-sequence pairs plus ridge regression targets for final object
displacement and rollout utility. This gives the experiments a trainable
learned-dynamics backend without adding a heavyweight ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from wam_inference_value.envs.block_push_2d import (
    BlockPush2D,
    BlockPushState,
    DynamicsParams,
    RolloutMetrics,
)


FEATURE_VERSION = 1
TARGET_NAMES = ("delta_x", "delta_y", "utility")


@dataclass(frozen=True)
class LearnedWamDataset:
    split: str
    mismatch: str
    features: np.ndarray
    targets: np.ndarray
    horizons: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    def summary(self) -> dict[str, Any]:
        utility = self.targets[:, 2] if self.targets.size else np.asarray([], dtype=float)
        return {
            "split": self.split,
            "mismatch": self.mismatch,
            "n_samples": self.n_samples,
            "feature_dim": int(self.features.shape[1]) if self.features.ndim == 2 else 0,
            "target_dim": int(self.targets.shape[1]) if self.targets.ndim == 2 else 0,
            "min_horizon": int(np.min(self.horizons)) if len(self.horizons) else None,
            "max_horizon": int(np.max(self.horizons)) if len(self.horizons) else None,
            "mean_utility": float(np.mean(utility)) if len(utility) else None,
            "std_utility": float(np.std(utility)) if len(utility) else None,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            split=np.asarray(self.split, dtype=object),
            mismatch=np.asarray(self.mismatch, dtype=object),
            features=self.features,
            targets=self.targets,
            horizons=self.horizons,
            metadata=np.asarray(json.dumps(self.metadata), dtype=object),
        )


@dataclass(frozen=True)
class LearnedPrediction:
    final_obj_xy: np.ndarray
    utility: float
    final_distance: float
    success: bool


@dataclass
class LearnedWamLiteModel:
    max_horizon: int
    episode_horizon: int
    x_mean: np.ndarray
    x_std: np.ndarray
    coef: np.ndarray
    ridge: float
    metadata: dict[str, Any]

    @classmethod
    def fit(
        cls,
        dataset: LearnedWamDataset,
        *,
        ridge: float = 1e-4,
        episode_horizon: int = 18,
    ) -> "LearnedWamLiteModel":
        x = np.asarray(dataset.features, dtype=float)
        y = np.asarray(dataset.targets, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("dataset features and targets must be 2D arrays with matching rows")
        if len(x) == 0:
            raise ValueError("cannot train learned WAM-lite on an empty dataset")

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x_norm = (x - x_mean[None, :]) / x_std[None, :]
        x_aug = np.column_stack([np.ones(len(x_norm), dtype=float), x_norm])

        penalty = float(ridge) * np.eye(x_aug.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = x_aug.T @ x_aug + penalty
        rhs = x_aug.T @ y
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(lhs) @ rhs

        utility = y[:, 2]
        utility_span = float(np.max(utility) - np.min(utility))
        metadata = dict(dataset.metadata)
        metadata.update(
            {
                "feature_version": FEATURE_VERSION,
                "target_names": list(TARGET_NAMES),
                "train_split": dataset.split,
                "train_mismatch": dataset.mismatch,
                "train_samples": int(len(x)),
                "utility_min": float(np.min(utility)),
                "utility_max": float(np.max(utility)),
                "utility_clip_margin": max(1.0, 0.25 * utility_span),
            }
        )
        return cls(
            max_horizon=int(metadata.get("max_horizon", np.max(dataset.horizons))),
            episode_horizon=int(episode_horizon),
            x_mean=x_mean,
            x_std=x_std,
            coef=coef,
            ridge=float(ridge),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            max_horizon=np.asarray(self.max_horizon, dtype=int),
            episode_horizon=np.asarray(self.episode_horizon, dtype=int),
            x_mean=self.x_mean,
            x_std=self.x_std,
            coef=self.coef,
            ridge=np.asarray(self.ridge, dtype=float),
            metadata=np.asarray(json.dumps(self.metadata), dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedWamLiteModel":
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            metadata_raw = data["metadata"].item()
            metadata = json.loads(str(metadata_raw))
            return cls(
                max_horizon=int(data["max_horizon"]),
                episode_horizon=int(data["episode_horizon"]),
                x_mean=np.asarray(data["x_mean"], dtype=float),
                x_std=np.asarray(data["x_std"], dtype=float),
                coef=np.asarray(data["coef"], dtype=float),
                ridge=float(data["ridge"]),
                metadata=metadata,
            )

    def _predict_targets_from_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.x_mean.shape[0]:
            raise ValueError(f"feature dimension {features.shape[1]} does not match model dimension {self.x_mean.shape[0]}")
        x_norm = (features - self.x_mean[None, :]) / self.x_std[None, :]
        x_aug = np.column_stack([np.ones(len(x_norm), dtype=float), x_norm])
        pred = x_aug @ self.coef
        margin = float(self.metadata.get("utility_clip_margin", 5.0))
        lo = float(self.metadata.get("utility_min", -20.0)) - margin
        hi = float(self.metadata.get("utility_max", 5.0)) + margin
        pred[:, 2] = np.clip(pred[:, 2], lo, hi)
        return pred

    def predict_batch(self, state: BlockPushState, actions_batch: np.ndarray) -> list[LearnedPrediction]:
        actions_batch = _as_action_batch(actions_batch)
        features = feature_matrix_for_state_actions(
            state=state,
            actions_batch=actions_batch,
            max_horizon=self.max_horizon,
            episode_horizon=self.episode_horizon,
        )
        targets = self._predict_targets_from_features(features)
        final_xy = np.asarray(state.obj_xy, dtype=float)[None, :] + targets[:, :2]
        target = np.asarray(state.target_xy, dtype=float)
        final_dist = np.linalg.norm(target[None, :] - final_xy, axis=1)
        return [
            LearnedPrediction(
                final_obj_xy=final_xy[i],
                utility=float(targets[i, 2]),
                final_distance=float(final_dist[i]),
                success=bool(final_dist[i] <= 0.33),
            )
            for i in range(len(targets))
        ]

    def predict_future_state_and_utility(
        self,
        env: BlockPush2D,
        state: BlockPushState,
        actions: np.ndarray,
    ) -> LearnedPrediction:
        pred = self.predict_batch(state, np.asarray(actions, dtype=float)[None, :, :])[0]
        success = pred.final_distance <= env.config.target_radius
        return LearnedPrediction(pred.final_obj_xy, pred.utility, pred.final_distance, bool(success))

    def predict_next_state_and_utility(
        self,
        env: BlockPush2D,
        state: BlockPushState,
        action: np.ndarray,
    ) -> LearnedPrediction:
        action_seq = np.asarray(action, dtype=float).reshape(1, 2)
        return self.predict_future_state_and_utility(env, state, action_seq)

    def predict_batch_metrics(
        self,
        env: BlockPush2D,
        state: BlockPushState,
        actions_batch: np.ndarray,
    ) -> list[RolloutMetrics]:
        actions_batch = _as_action_batch(actions_batch)
        predictions = self.predict_batch(state, actions_batch)
        initial_distance = env.distance_to_target(state)
        energy = np.sum(actions_batch * actions_batch, axis=(1, 2))
        out: list[RolloutMetrics] = []
        for i, pred in enumerate(predictions):
            final_distance = float(np.linalg.norm(np.asarray(state.target_xy, dtype=float) - pred.final_obj_xy))
            success = final_distance <= env.config.target_radius
            progress = float(initial_distance - final_distance)
            excess = max(0.0, float(np.linalg.norm(pred.final_obj_xy)) - env.config.workspace_radius)
            out.append(
                RolloutMetrics(
                    initial_distance=float(initial_distance),
                    final_distance=final_distance,
                    progress=progress,
                    energy=float(energy[i]),
                    safety_violation=float(excess * excess),
                    success=bool(success),
                    utility=float(pred.utility),
                )
            )
        return out

    def evaluate(self, dataset: LearnedWamDataset) -> dict[str, Any]:
        pred = self._predict_targets_from_features(dataset.features)
        target = np.asarray(dataset.targets, dtype=float)
        delta_error = pred[:, :2] - target[:, :2]
        utility_error = pred[:, 2] - target[:, 2]
        corr = np.nan
        if len(target) > 1 and np.std(pred[:, 2]) > 1e-12 and np.std(target[:, 2]) > 1e-12:
            corr = float(np.corrcoef(pred[:, 2], target[:, 2])[0, 1])
        return {
            "split": dataset.split,
            "mismatch": dataset.mismatch,
            "n_samples": dataset.n_samples,
            "final_delta_mae": float(np.mean(np.abs(delta_error))),
            "final_position_l2_mae": float(np.mean(np.linalg.norm(delta_error, axis=1))),
            "utility_mae": float(np.mean(np.abs(utility_error))),
            "utility_rmse": float(np.sqrt(np.mean(utility_error * utility_error))),
            "utility_corr": corr,
        }


def _as_action_batch(actions_batch: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions_batch, dtype=float)
    if actions.ndim != 3 or actions.shape[2] != 2:
        raise ValueError("actions_batch must have shape (n_rollouts, horizon, 2)")
    return actions


def feature_matrix_for_state_actions(
    *,
    state: BlockPushState,
    actions_batch: np.ndarray,
    max_horizon: int,
    episode_horizon: int,
) -> np.ndarray:
    actions = _as_action_batch(actions_batch)
    n, horizon, _ = actions.shape
    max_horizon = int(max_horizon)
    if horizon > max_horizon:
        raise ValueError(f"action horizon {horizon} exceeds learned model max_horizon {max_horizon}")

    padded = np.zeros((n, max_horizon, 2), dtype=float)
    padded[:, :horizon, :] = actions
    flat_actions = padded.reshape(n, max_horizon * 2)

    obj = np.asarray(state.obj_xy, dtype=float)
    target = np.asarray(state.target_xy, dtype=float)
    to_goal = target - obj
    dist = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / dist if dist > 1e-12 else np.array([1.0, 0.0])
    perp = np.array([-goal_dir[1], goal_dir[0]])

    norms = np.linalg.norm(actions, axis=2)
    action_sum = actions.sum(axis=1)
    action_mean = actions.mean(axis=1)
    first_action = actions[:, 0, :]
    last_action = actions[:, -1, :]
    energy = np.sum(actions * actions, axis=(1, 2))
    mean_norm = norms.mean(axis=1)
    max_norm = norms.max(axis=1)
    forward = actions @ goal_dir
    lateral = actions @ perp
    forward_sum = forward.sum(axis=1)
    lateral_abs_sum = np.abs(lateral).sum(axis=1)

    state_features = np.column_stack(
        [
            np.repeat(obj[None, :], n, axis=0),
            np.repeat(target[None, :], n, axis=0),
            np.repeat(to_goal[None, :], n, axis=0),
            np.full(n, dist, dtype=float),
            np.full(n, state.t / max(1, episode_horizon), dtype=float),
            np.full(n, horizon / max(1, max_horizon), dtype=float),
        ]
    )
    action_features = np.column_stack(
        [
            flat_actions,
            action_sum,
            action_mean,
            first_action,
            last_action,
            energy,
            mean_norm,
            max_norm,
            forward_sum,
            lateral_abs_sum,
        ]
    )
    base = np.column_stack([state_features, action_features])
    interactions = np.column_stack(
        [
            action_sum[:, 0] * to_goal[0],
            action_sum[:, 1] * to_goal[1],
            energy * dist,
            forward_sum / max(1, horizon),
            lateral_abs_sum / max(1, horizon),
        ]
    )
    return np.column_stack([base, base * base, interactions])


def rollout_batch_final_positions(
    env: BlockPush2D,
    state: BlockPushState,
    actions_batch: np.ndarray,
    params: DynamicsParams,
    *,
    use_nonstationary_shift: bool = True,
) -> np.ndarray:
    actions = _as_action_batch(actions_batch)
    n, horizon, _ = actions.shape
    pos = np.repeat(np.asarray(state.obj_xy, dtype=float)[None, :], n, axis=0)
    for h in range(horizon):
        p = params
        if use_nonstationary_shift:
            p = env.shifted_params(params, state.t + h)
        action = actions[:, h, :]
        mag = np.linalg.norm(action, axis=1)
        scale = np.ones(n, dtype=float)
        over = mag > env.config.max_push
        scale[over] = env.config.max_push / np.maximum(mag[over], 1e-12)
        action = action * scale[:, None]
        mag = np.linalg.norm(action, axis=1)
        direction = np.zeros_like(action)
        active = mag > 1e-12
        direction[active] = action[active] / mag[active, None]
        force = np.minimum(mag, env.config.max_push)
        release = np.ones(n, dtype=float)
        if p.stuck > 0.0:
            release[force < env.config.release_threshold] = 0.0
        stuck_scale = max(0.0, 1.0 - p.stuck)
        base_gain = max(0.02, 1.0 - p.friction) / max(0.2, p.mass)
        slip_loss = np.maximum(0.0, 1.0 - p.slip * force * force)
        step_vec = direction * (force * base_gain * slip_loss * stuck_scale * release)[:, None]
        if p.lateral_slip > 0.0:
            perp = np.column_stack([-direction[:, 1], direction[:, 0]])
            sign = 1.0 if ((state.t + h) % 2 == 0) else -1.0
            step_vec += sign * perp * (p.lateral_slip * force * force)[:, None]
        pos += step_vec
    return pos


def generate_blockpush_dataset(
    *,
    n_states: int,
    rollouts_per_state: int,
    mismatch: str,
    seed: int,
    split: str,
    env: BlockPush2D | None = None,
    max_horizon: int | None = None,
    variable_horizons: bool = True,
) -> LearnedWamDataset:
    from wam_inference_value.rollouts import sample_action_sequences

    env = env or BlockPush2D()
    max_horizon = int(max_horizon or env.config.horizon)
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    horizons_all: list[np.ndarray] = []

    for state_id in range(int(n_states)):
        state_seed = int(seed + 7919 * state_id + 17)
        state = env.sample_state(state_seed, mismatch=mismatch, state_id=state_id)
        action_seed = int(seed + 104_729 * (state_id + 1))
        actions_full = sample_action_sequences(env, state, rollouts_per_state, max_horizon, action_seed)
        if variable_horizons:
            horizons = rng.integers(1, max_horizon + 1, size=int(rollouts_per_state))
        else:
            horizons = np.full(int(rollouts_per_state), max_horizon, dtype=int)

        for horizon in sorted(set(int(h) for h in horizons)):
            idx = np.flatnonzero(horizons == horizon)
            actions = actions_full[idx, :horizon, :]
            metrics = env.rollout_batch_metrics(state, actions, state.true_params, use_nonstationary_shift=True)
            final_positions = rollout_batch_final_positions(
                env,
                state,
                actions,
                state.true_params,
                use_nonstationary_shift=True,
            )
            x = feature_matrix_for_state_actions(
                state=state,
                actions_batch=actions,
                max_horizon=max_horizon,
                episode_horizon=env.config.episode_horizon,
            )
            y = np.column_stack(
                [
                    final_positions - np.asarray(state.obj_xy, dtype=float)[None, :],
                    np.asarray([m.utility for m in metrics], dtype=float),
                ]
            )
            features.append(x)
            targets.append(y)
            horizons_all.append(np.full(len(idx), horizon, dtype=int))

    feature_arr = np.vstack(features) if features else np.zeros((0, 0), dtype=float)
    target_arr = np.vstack(targets) if targets else np.zeros((0, len(TARGET_NAMES)), dtype=float)
    horizon_arr = np.concatenate(horizons_all) if horizons_all else np.zeros(0, dtype=int)
    metadata = {
        "dataset": "BlockPush2D",
        "split": split,
        "mismatch": mismatch,
        "seed": int(seed),
        "n_states": int(n_states),
        "rollouts_per_state": int(rollouts_per_state),
        "max_horizon": int(max_horizon),
        "variable_horizons": bool(variable_horizons),
        "feature_version": FEATURE_VERSION,
    }
    return LearnedWamDataset(split=split, mismatch=mismatch, features=feature_arr, targets=target_arr, horizons=horizon_arr, metadata=metadata)


def train_learned_wam_lite(
    *,
    model_path: str | Path | None,
    dataset_dir: str | Path | None = None,
    id_mismatch: str = "mild",
    seed: int = 101,
    train_states: int = 64,
    train_rollouts: int = 96,
    val_states: int = 24,
    val_rollouts: int = 96,
    ood_states: int = 16,
    ood_rollouts: int = 64,
    max_horizon: int = 12,
    ridge: float = 1e-4,
    ood_mismatches: tuple[str, ...] = ("severe", "stuck_slip", "nonstationary"),
) -> tuple[LearnedWamLiteModel, dict[str, Any]]:
    env = BlockPush2D()
    train = generate_blockpush_dataset(
        n_states=train_states,
        rollouts_per_state=train_rollouts,
        mismatch=id_mismatch,
        seed=seed,
        split="train",
        env=env,
        max_horizon=max_horizon,
        variable_horizons=True,
    )
    val = generate_blockpush_dataset(
        n_states=val_states,
        rollouts_per_state=val_rollouts,
        mismatch=id_mismatch,
        seed=seed + 1_000_003,
        split="validation",
        env=env,
        max_horizon=max_horizon,
        variable_horizons=True,
    )
    model = LearnedWamLiteModel.fit(train, ridge=ridge, episode_horizon=env.config.episode_horizon)
    dataset_artifacts: dict[str, str] = {}
    if dataset_dir is not None:
        dataset_path = Path(dataset_dir)
        train_path = dataset_path / "learned_wam_lite_train.npz"
        val_path = dataset_path / "learned_wam_lite_validation.npz"
        train.save(train_path)
        val.save(val_path)
        dataset_artifacts["train"] = str(train_path)
        dataset_artifacts["validation"] = str(val_path)
    metrics = {
        "train": model.evaluate(train),
        "validation": model.evaluate(val),
        "ood": [],
    }
    dataset_summaries = [train.summary(), val.summary()]
    for i, mismatch in enumerate(ood_mismatches):
        ood = generate_blockpush_dataset(
            n_states=ood_states,
            rollouts_per_state=ood_rollouts,
            mismatch=mismatch,
            seed=seed + 2_000_003 + 997 * i,
            split="ood",
            env=env,
            max_horizon=max_horizon,
            variable_horizons=True,
        )
        metrics["ood"].append(model.evaluate(ood))
        dataset_summaries.append(ood.summary())
        if dataset_dir is not None:
            ood_path = Path(dataset_dir) / f"learned_wam_lite_ood_{mismatch}.npz"
            ood.save(ood_path)
            dataset_artifacts[f"ood_{mismatch}"] = str(ood_path)

    if model_path is not None:
        model.save(model_path)
    summary = {
        "model": "learned_wam_lite_ridge",
        "model_path": str(model_path) if model_path is not None else None,
        "id_mismatch": id_mismatch,
        "seed": int(seed),
        "ridge": float(ridge),
        "max_horizon": int(max_horizon),
        "datasets": dataset_summaries,
        "dataset_artifacts": dataset_artifacts,
        "metrics": metrics,
        "metadata": model.metadata,
    }
    return model, summary


def load_or_train_learned_wam_lite(
    *,
    model_path: str | Path,
    train_if_missing: bool,
    id_mismatch: str = "mild",
    seed: int = 101,
    train_states: int = 64,
    train_rollouts: int = 96,
    val_states: int = 24,
    val_rollouts: int = 96,
    max_horizon: int = 12,
) -> LearnedWamLiteModel:
    path = Path(model_path)
    if path.exists():
        return LearnedWamLiteModel.load(path)
    if not train_if_missing:
        raise FileNotFoundError(f"learned WAM-lite model not found: {path}")
    model, _ = train_learned_wam_lite(
        model_path=path,
        id_mismatch=id_mismatch,
        seed=seed,
        train_states=train_states,
        train_rollouts=train_rollouts,
        val_states=val_states,
        val_rollouts=val_rollouts,
        max_horizon=max_horizon,
    )
    return model
