import numpy as np
import torch
import lightgbm as lgb
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from basicts.metrics import ALL_METRICS
import joblib
import json
import os

@dataclass
class LGBMConfig:
    input_len: int = 12
    output_len: int = 12
    use_tod: bool = False
    use_dow: bool = False
    n_estimators: int = 300
    learning_rate: float = 0.05
    min_child_samples: int = 100
    num_leaves: int = 15
    max_depth: int = 5
    patience: int = 30
    horizons: tuple = (3, 6, 12)
    max_bin: int = 63
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    min_gain_to_split: float = 0.0

class LGBM:
    """One LightGBM per horizon, shared across sensors (each sensor window is an example).
    Missing values (0.0) in the input are used as is, missing targets are dropped."""

    def __init__(self, config: LGBMConfig):
        self.config = config
        self.use_time = self.config.use_dow or self.config.use_tod
    def windows(self, path, split):
        c = self.config
        data = np.load(f"{path}/{split}_data.npy").astype(np.float32)  # (T, N)

        w = sliding_window_view(data, c.input_len + c.output_len, axis=0)  # (T', N, L + H)
        S = w.shape[1]
        X = w[..., :c.input_len].reshape(-1, c.input_len).copy()
        Y = w[..., c.input_len:].reshape(-1, c.output_len).copy()

        if self.use_time:
            timestamps = np.load(f"{path}/{split}_timestamps.npy").astype(np.float32) # (T, 2)
            encoded = []
            if self.config.use_tod:
                encoded.append(np.sin(2 * np.pi * timestamps[:, 0]))
                encoded.append(np.cos(2 * np.pi * timestamps[:, 0]))
            if self.config.use_dow:
                encoded.append(np.sin(2 * np.pi * timestamps[:, 1]))
                encoded.append(np.cos(2 * np.pi * timestamps[:, 1]))
            timestamps = np.stack(encoded, axis=-1)
            ts = sliding_window_view(timestamps, c.input_len + c.output_len, axis=0) # (T', 2/4, L + H)
            X_ts = ts[..., :c.input_len].copy()
            T, F, L = X_ts.shape
            X_ts = np.permute_dims(X_ts, (0, 2, 1)).reshape((T, 1, L, F)) 
            X_ts = np.broadcast_to(X_ts, (T, S, L, F)).reshape(T * S, L * F)
            X = np.concat([X, X_ts], axis=-1)
        return X, Y

    def fit(self, X, Y, X_val, Y_val):
        c = self.config
        self.models = []
        for h in range(len(self.config.horizons)):
            horizon = self.config.horizons[h] - 1
            m, m_val = Y[:, horizon] != 0, Y_val[:, horizon] != 0
            print(f"Horizon {horizon + 1}:\n")
            model = lgb.LGBMRegressor(
                objective="l1", n_estimators=c.n_estimators, learning_rate=c.learning_rate,
                num_leaves=c.num_leaves, max_depth=c.max_depth, min_child_samples=c.min_child_samples,
                n_jobs=-1, force_col_wise=True, max_bin=c.max_bin, device_type="cpu",
                reg_alpha=c.lambda_l1, reg_lambda=c.lambda_l2, min_split_gain=c.min_gain_to_split,
                bagging_fraction=c.bagging_fraction, bagging_freq=c.bagging_freq, feature_fraction=c.feature_fraction,
                random_state=42
            )   
            model.fit(
                X[m], Y[m, horizon],
                eval_X=X_val[m_val], eval_y=Y_val[m_val, horizon],
                callbacks=[lgb.early_stopping(c.patience, verbose=False), lgb.log_evaluation(10)],
            )
            self.models.append(model)

    def predict(self, X):
        return np.stack([model.predict(X) for model in self.models], axis=1)

    def evaluate(self, X, Y):
        pred, target = torch.tensor(self.predict(X)), torch.tensor(Y)
        mask = target != 0
        results = {}
        for h in range(len(self.config.horizons)):
            horizon = self.config.horizons[h] - 1
            results[f"h{horizon + 1}"] = {
                name: ALL_METRICS[name](pred[:, h], target[:, horizon], mask[:, horizon]).item()
                for name in ("MAE", "RMSE", "MAPE")
            }
        results["all"] = {name: ALL_METRICS[name](pred, target[:, [h - 1 for h in self.config.horizons]], mask[:, [h - 1 for h in self.config.horizons]]).item() for name in ("MAE", "RMSE", "MAPE")}
        return results

    def run(self, path="data/METR-LA", folder_name: str = None):
        self.fit(*self.windows(path, "train"), *self.windows(path, "val"))
        results = self.evaluate(*self.windows(path, "test"))
        for k, v in results.items():
            print(k, v)

        if folder_name is None:
            folder_name = "checkpoints/LGBM/" + f"input_{self.config.input_len}" + ("_tod" if self.config.use_tod else "") + ("_dow" if self.config.use_dow else "")
        os.makedirs(f"{folder_name}")
        joblib.dump(self.models, f"{folder_name}/models.pkl")
        with open(f"{folder_name}/results.json", "w") as f:
            json.dump(results, f, indent=4)
        return results
