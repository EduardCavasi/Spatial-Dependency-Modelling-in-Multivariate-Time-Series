import json
import os
from typing import List, Tuple, Dict
import numpy as np
import torch
import lightgbm as lgb
from torch import nn
import joblib

from basicts.configs import BasicTSModelConfig
from basicts.scaler import BasicTSScaler

from src.utils.scaler.z_score_scaler import MyZScoreScaler

def load_nn_checkpoint(
    path: str, # path to the run folder containing the best model weights
    model_cls: type[nn.Module],
    model_config: BasicTSModelConfig,
    scaler_cls: type[BasicTSScaler],
    scaler_config: Dict
) -> Tuple[nn.Module, BasicTSScaler]:
    filePath = None
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith("best_val_MAE.pt"):
                filePath = os.path.join(root, file)
    if filePath is None:
        raise FileNotFoundError

    ckpt = torch.load(
        filePath,
        map_location="cpu",
        weights_only=False,
    )

    model = model_cls(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = scaler_cls(
        **scaler_config,
        stats = ckpt["scaler_stats"]
    )
    return model, scaler

def load_lgbm_checkpoint(
    path: str, # path to the run folder containing the best model weights
) -> List[lgb.LGBMModel]:
    filePath = None
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".pkl"):
                filePath = os.path.join(root, file)
    if filePath is None:
        raise FileNotFoundError

    models = joblib.load(filePath)
    return models

def load_test_results(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    results_dir = os.path.join(path, "test_results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(results_dir)

    with open(os.path.join(path, "cfg.json")) as f:
        params = json.load(f)["dataset_params"]
    n_sensors = np.load(os.path.join(params["data_file_path"], "test_data.npy"), mmap_mode="r").shape[1]

    def load_array(name, length):
        values = np.fromfile(os.path.join(results_dir, name), dtype=np.float32)
        return values.reshape(-1, length, n_sensors)

    return (
        load_array("inputs.npy", params["input_len"]),
        load_array("prediction.npy", params["output_len"]),
        load_array("targets.npy", params["output_len"]),
    )