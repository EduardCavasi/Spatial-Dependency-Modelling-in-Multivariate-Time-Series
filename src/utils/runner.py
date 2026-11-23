from basicts.utils.constants import BasicTSMode
from basicts import BasicTSLauncher
from src.model.baseline_temporal_models.encoder_decoder_gru import EncoderDecoderGRU
from src.model.baseline_temporal_models.mlp import MLP, MLPConfig
from src.model.baseline_temporal_models.gru import GRU, GRUConfig
from src.model.baseline_temporal_models.lgbm import LGBM, LGBMConfig
from src.model.baseline_temporal_models.encoder_decoder_gru import EncoderDecoderGRU, EncoderDecoderGRUConfig

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
    },
    "EncoderDecoderGRU": {
        "model": EncoderDecoderGRU,
        "model_config": EncoderDecoderGRUConfig
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
parser.add_argument("--tod", action="store_true")
parser.add_argument("--dow", action="store_true")
parser.add_argument("--input", type=int, default=12)
args = parser.parse_args()

if args.model == "LGBM":
    LGBM(LGBMConfig(
        input_len=args.input,
        use_tod=args.tod,
        use_dow=args.dow
    )).run()
else:
    cfg = make_cfg(
        model = MODELS[args.model]["model"],
        model_config = MODELS[args.model]["model_config"](use_tod=args.tod, use_dow=args.dow, input_len = args.input),
        num_epochs = 10,
        use_timestamps = args.tod or args.dow,
        input_len = args.input
    )
    BasicTSLauncher.launch_training(cfg)