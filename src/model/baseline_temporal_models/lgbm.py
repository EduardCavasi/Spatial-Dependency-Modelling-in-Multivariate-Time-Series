import numpy as np
import torch
import lightgbm as lgb
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from basicts.metrics import ALL_METRICS


@dataclass
class LGBMConfig:
    input_len: int = 12
    output_len: int = 12
    n_estimators: int = 300
    learning_rate: float = 0.05
    min_child_samples: int = 100
    num_leaves: int = 15
    max_depth: int = 5
    patience: int = 30
    horizons: tuple = (3, 6, 12)
    max_bin: int = 63


class LGBM:
    """One LightGBM per horizon, shared across sensors (each sensor window is an example).
    Missing values (0.0) in the inputs are NaN (LightGBM handles them), missing targets are dropped."""

    def __init__(self, config: LGBMConfig):
        self.config = config

    def windows(self, path, split):
        c = self.config
        data = np.load(f"{path}/{split}_data.npy").astype(np.float32)  # (T, N)
        w = sliding_window_view(data, c.input_len + c.output_len, axis=0)  # (T', N, L + H)
        X = w[..., :c.input_len].reshape(-1, c.input_len).copy()
        Y = w[..., c.input_len:].reshape(-1, c.output_len).copy()
        return X, Y

    def fit(self, X, Y, X_val, Y_val):
        c = self.config
        self.models = []
        for h in range(c.output_len):
            m, m_val = Y[:, h] != 0, Y_val[:, h] != 0
            print(f"Horizon {h}:\n")
            model = lgb.LGBMRegressor(
                objective="l1", n_estimators=c.n_estimators, learning_rate=c.learning_rate,
                num_leaves=c.num_leaves, max_depth=c.max_depth, min_child_samples=c.min_child_samples,
                n_jobs=-1, force_col_wise=True, max_bin=c.max_bin
            )
            model.fit(
                X[m], Y[m, h],
                eval_X=X_val[m_val], eval_y=Y_val[m_val, h],
                callbacks=[lgb.early_stopping(c.patience, verbose=False), lgb.log_evaluation(10)],
            )
            self.models.append(model)

    def predict(self, X):
        return np.stack([model.predict(X) for model in self.models], axis=1)

    def evaluate(self, X, Y):
        pred, target = torch.tensor(self.predict(X)), torch.tensor(Y)
        mask = target != 0
        results = {}
        for h in self.config.horizons:
            results[f"h{h}"] = {
                name: ALL_METRICS[name](pred[:, h - 1], target[:, h - 1], mask[:, h - 1]).item()
                for name in ("MAE", "RMSE", "MAPE")
            }
        results["all"] = {name: ALL_METRICS[name](pred, target, mask).item() for name in ("MAE", "RMSE", "MAPE")}
        return results

    def run(self, path="data/METR-LA"):
        self.fit(*self.windows(path, "train"), *self.windows(path, "val"))
        results = self.evaluate(*self.windows(path, "test"))
        for k, v in results.items():
            print(k, v)
        return results
