from basicts.utils.constants import BasicTSMode
from basicts import BasicTSLauncher
from src.model.baseline_temporal_models.mlp import MLP, MLPConfig
from src.model.baseline_temporal_models.gru import GRU, GRUConfig
from src.model.baseline_temporal_models.lgbm import LGBM, LGBMConfig

from src.utils.config import make_cfg
import argparse

MODELS = {
    "MLP": {
        "model": MLP,
        "model_config": MLPConfig
    },
    "GRU": {
        "model": GRU,
        "model_config": GRUConfig
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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=sorted(MODELS) + ["LGBM"], default="MLP")

args = parser.parse_args()

if args.model == "LGBM":
    LGBM(LGBMConfig()).run()
else:
    cfg = make_cfg(**MODELS[args.model], num_epochs = 10)
    BasicTSLauncher.launch_training(cfg)