"""Small NumPy WAM-lite backbones used by max-out experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WAMDataset:
    x: np.ndarray
    y: np.ndarray
    metadata: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=self.x, y=self.y, metadata=np.asarray(json.dumps(self.metadata), dtype=object))


class RidgeRegressor:
    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(ridge)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean = x.mean(axis=0)
        self.x_std = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
        xn = (x - self.x_mean[None, :]) / self.x_std[None, :]
        xa = np.column_stack([np.ones(len(xn)), xn])
        penalty = self.ridge * np.eye(xa.shape[1])
        penalty[0, 0] = 0.0
        lhs = xa.T @ xa + penalty
        rhs = xa.T @ y
        try:
            self.coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None or self.coef is None:
            raise ValueError("model is not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        xn = (x - self.x_mean[None, :]) / self.x_std[None, :]
        xa = np.column_stack([np.ones(len(xn)), xn])
        return xa @ self.coef

    def state_dict(self) -> dict:
        return {"ridge": self.ridge, "x_mean": self.x_mean, "x_std": self.x_std, "coef": self.coef}

    @classmethod
    def from_state_dict(cls, state: dict) -> "RidgeRegressor":
        out = cls(float(state["ridge"]))
        out.x_mean = np.asarray(state["x_mean"], dtype=float)
        out.x_std = np.asarray(state["x_std"], dtype=float)
        out.coef = np.asarray(state["coef"], dtype=float)
        return out


class HorizonWAM:
    """Ridge model from state/action sequence to final state delta and utility."""

    name = "horizon_wam"

    def __init__(self, ridge: float = 1e-4) -> None:
        self.regressor = RidgeRegressor(ridge)

    def fit(self, dataset: WAMDataset) -> "HorizonWAM":
        self.regressor.fit(dataset.x, dataset.y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.regressor.predict(x)

    def evaluate(self, dataset: WAMDataset) -> dict:
        pred = self.predict(dataset.x)
        err = pred - dataset.y
        corr = np.nan
        if len(pred) > 1 and np.std(pred[:, -1]) > 1e-12 and np.std(dataset.y[:, -1]) > 1e-12:
            corr = float(np.corrcoef(pred[:, -1], dataset.y[:, -1])[0, 1])
        return {
            "model": self.name,
            "n_samples": int(len(dataset.x)),
            "state_delta_mae": float(np.mean(np.abs(err[:, :-1]))),
            "utility_mae": float(np.mean(np.abs(err[:, -1]))),
            "utility_corr": corr,
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        state = self.regressor.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **state, metadata=np.asarray(json.dumps(metadata or {}), dtype=object), model=np.asarray(self.name, dtype=object))


class MLPDynamicsWAM:
    """One-hidden-layer MLP trained by deterministic full-batch gradient descent."""

    name = "mlp_dynamics_wam"

    def __init__(self, hidden_dim: int = 32, lr: float = 0.015, epochs: int = 350, seed: int = 0) -> None:
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.params: dict[str, np.ndarray] = {}
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def fit(self, dataset: WAMDataset) -> "MLPDynamicsWAM":
        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)
        self.x_mean = x.mean(axis=0)
        self.x_std = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
        self.y_mean = y.mean(axis=0)
        self.y_std = np.where(y.std(axis=0) < 1e-8, 1.0, y.std(axis=0))
        x = (x - self.x_mean[None, :]) / self.x_std[None, :]
        y = (y - self.y_mean[None, :]) / self.y_std[None, :]
        rng = np.random.default_rng(self.seed)
        w1 = rng.normal(0.0, 0.18, size=(x.shape[1], self.hidden_dim))
        b1 = np.zeros(self.hidden_dim)
        w2 = rng.normal(0.0, 0.18, size=(self.hidden_dim, y.shape[1]))
        b2 = np.zeros(y.shape[1])
        for _ in range(self.epochs):
            h_pre = x @ w1 + b1
            h = np.tanh(h_pre)
            pred = h @ w2 + b2
            grad = 2.0 * (pred - y) / len(x)
            gw2 = h.T @ grad
            gb2 = grad.sum(axis=0)
            gh = grad @ w2.T
            gz = gh * (1.0 - h * h)
            gw1 = x.T @ gz
            gb1 = gz.sum(axis=0)
            w1 -= self.lr * gw1
            b1 -= self.lr * gb1
            w2 -= self.lr * gw2
            b2 -= self.lr * gb2
        self.params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
            raise ValueError("model is not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        xn = (x - self.x_mean[None, :]) / self.x_std[None, :]
        h = np.tanh(xn @ self.params["w1"] + self.params["b1"])
        yn = h @ self.params["w2"] + self.params["b2"]
        return yn * self.y_std[None, :] + self.y_mean[None, :]

    def evaluate(self, dataset: WAMDataset) -> dict:
        pred = self.predict(dataset.x)
        err = pred - dataset.y
        corr = np.nan
        if len(pred) > 1 and np.std(pred[:, -1]) > 1e-12 and np.std(dataset.y[:, -1]) > 1e-12:
            corr = float(np.corrcoef(pred[:, -1], dataset.y[:, -1])[0, 1])
        return {
            "model": self.name,
            "n_samples": int(len(dataset.x)),
            "state_delta_mae": float(np.mean(np.abs(err[:, :-1]))),
            "utility_mae": float(np.mean(np.abs(err[:, -1]))),
            "utility_corr": corr,
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **self.params,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            metadata=np.asarray(json.dumps(metadata or {}), dtype=object),
            model=np.asarray(self.name, dtype=object),
        )


class EnsembleWAM:
    """Bootstrap ensemble of horizon ridge WAMs."""

    name = "ensemble_wam"

    def __init__(self, n_models: int = 5, ridge: float = 1e-4, seed: int = 0) -> None:
        self.n_models = int(n_models)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.models: list[HorizonWAM] = []

    def fit(self, dataset: WAMDataset) -> "EnsembleWAM":
        rng = np.random.default_rng(self.seed)
        self.models = []
        for _ in range(self.n_models):
            idx = rng.integers(0, len(dataset.x), size=len(dataset.x))
            sub = WAMDataset(dataset.x[idx], dataset.y[idx], dataset.metadata)
            self.models.append(HorizonWAM(self.ridge).fit(sub))
        return self

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(x) for m in self.models], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.mean(self.predict_members(x), axis=0)

    def uncertainty(self, x: np.ndarray) -> np.ndarray:
        return np.var(self.predict_members(x), axis=0)

    def evaluate(self, dataset: WAMDataset) -> dict:
        pred = self.predict(dataset.x)
        err = pred - dataset.y
        utility_var = self.uncertainty(dataset.x)[:, -1]
        return {
            "model": self.name,
            "n_samples": int(len(dataset.x)),
            "state_delta_mae": float(np.mean(np.abs(err[:, :-1]))),
            "utility_mae": float(np.mean(np.abs(err[:, -1]))),
            "mean_utility_variance": float(np.mean(utility_var)),
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": np.asarray(json.dumps(metadata or {}), dtype=object), "model": np.asarray(self.name, dtype=object)}
        for i, model in enumerate(self.models):
            state = model.regressor.state_dict()
            for key, value in state.items():
                payload[f"m{i}_{key}"] = value
        np.savez(path, **payload)
