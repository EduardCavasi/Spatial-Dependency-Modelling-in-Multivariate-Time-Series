from src.model.baseline_temporal_models.encoder_decoder_gru import EncoderDecoderGRU
from src.model.baseline_temporal_models.mlp import MLP, MLPConfig
from src.model.baseline_temporal_models.gru import GRU, GRUConfig
from src.model.baseline_temporal_models.lgbm import LGBM, LGBMConfig
from src.model.baseline_temporal_models.encoder_decoder_gru import EncoderDecoderGRU, EncoderDecoderGRUConfig
from src.model.baseline_temporal_models.linear import Linear, LinearConfig
from src.model.baseline_temporal_models.lstm import LSTM, LSTMConfig
from src.model.baseline_temporal_models.tcn import TCN, TCNConfig
from src.model.baseline_temporal_models.encoder_decoder_lstm import EncoderDecoderLSTM, EncoderDecoderLSTMConfig

from src.utils.config import make_cfg

import argparse
from typing import Optional, List
from basicts.utils.constants import BasicTSMode
from basicts import BasicTSLauncher
import random as rand
import os
import json
import numpy as np
import shutil

MODELS = {
    "MLP": {
        "model": MLP,
        "model_config": MLPConfig
    },
    "GRU": {
        "model": GRU,
        "model_config": GRUConfig
    },
    "EncoderDecoderGRU": {
        "model": EncoderDecoderGRU,
        "model_config": EncoderDecoderGRUConfig
    },
    "Linear": {
        "model": Linear,
        "model_config": LinearConfig
    },
    "LSTM": {
        "model": LSTM,
        "model_config": LSTMConfig
    },
    "TCN": {
        "model": TCN,
        "model_config": TCNConfig
    },
    "EncoderDecoderLSTM": {
        "model": EncoderDecoderLSTM,
        "model_config": EncoderDecoderLSTMConfig
    }
}

dataset_params = {
    "dataset_name": "METR-LA",
    "input_len": 12,
    "output_len": 12,
    "mode": BasicTSMode.TRAIN,
    "use_timestamps": False,
    "local": True,
    "data_file_path": "data/METR-LA",
    "memmap": False
}

ALIAS = {
    "all": ("overall", "all"),
    "h3": ("horizon_3", "h3"),
    "h6": ("horizon_6", "h6"),
    "h12": ("horizon_12", "h12"),
}

def pick_metric(metrics, name):
    for key in ALIAS[name]:
        if key in metrics:
            return metrics[key]
    raise KeyError(name)

def run(args: argparse.Namespace, random: bool = False, save_results: bool = True) -> Optional[List]:
    if args.model == "LGBM":
        metrics = LGBM(LGBMConfig(
            input_len=args.input,
            use_tod=args.tod,
            use_dow=args.dow,
            save_results=save_results,
            random=random
        )).run()
    else:
        params = {
            "use_tod": args.tod,
            "use_dow": args.dow,
            "input_len": args.input
        }
        if args.all_sensors and args.model == "EncoderDecoderGRU":
            params["input_size"] = 207
        cfg = make_cfg(
            model = MODELS[args.model]["model"],
            model_config = MODELS[args.model]["model_config"](**params),
            num_epochs = args.epochs,
            use_timestamps = args.tod or args.dow,
            input_len = args.input,
            norm_each_channel = args.norm_each_channel,
            save_results = save_results,
            seed=42 if not random else rand.randint(1, 100)
        )
        BasicTSLauncher.launch_training(cfg)

        run_dir = os.path.join(cfg.ckpt_save_dir, cfg.md5)
        with open(os.path.join(run_dir, "test_metrics.json")) as f:
            metrics = json.load(f)

    if not save_results:
        run_dir = os.path.join(cfg.ckpt_save_dir, cfg.md5)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS) + ["LGBM"], default="MLP")
    parser.add_argument("--tod", action="store_true")
    parser.add_argument("--dow", action="store_true")
    parser.add_argument("--all-sensors", action="store_true")
    parser.add_argument("--input", type=int, default=12)
    parser.add_argument("--runtimes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping may end sooner)")
    parser.add_argument("--norm-each-channel", action="store_true")
    args = parser.parse_args()

    if args.runtimes == 1:
        run(args)
    else:
        h3 = []
        h6 = []
        h12 = []
        all = []
        for _ in range(args.runtimes):
            metrics = run(args, random=True, save_results = False)
            h3.append(pick_metric(metrics, "h3")["MAE"])
            h6.append(pick_metric(metrics, "h6")["MAE"])
            h12.append(pick_metric(metrics, "h12")["MAE"])
            all.append(pick_metric(metrics, "all")["MAE"])

        print(f"H3 -> mean: {np.mean(h3)}, std: {np.std(h3)}")
        print(f"H6 -> mean: {np.mean(h6)}, std: {np.std(h6)}")
        print(f"H12 -> mean: {np.mean(h12)}, std: {np.std(h12)}")
        print(f"ALL -> mean: {np.mean(all)}, std: {np.std(all)}")
