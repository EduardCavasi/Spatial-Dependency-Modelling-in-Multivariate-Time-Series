from basicts.configs import BasicTSForecastingConfig
from basicts.data import BasicTSForecastingDataset
from basicts.runners.callback import EarlyStopping
from src.utils.taskflow.dcrnn_taskflow import DCRNNTaskFlow
from basicts.utils.constants import BasicTSMode

from src.utils.scaler.z_score_scaler import MyZScoreScaler
from torch.optim.lr_scheduler import MultiStepLR

def make_cfg(model, model_config, **overrides):
    cfg = dict(
        dataset_type=BasicTSForecastingDataset,
        dataset_params={
            "dataset_name": "METR-LA",
            "mode": BasicTSMode.TRAIN,
            "local": True,
            "data_file_path": "data/METR-LA",
            "memmap": False,
        },
        input_len=12,
        output_len=12,
        use_timestamps=False,
        null_val=0.0,
        null_to_num=0.0,
        scaler=MyZScoreScaler,
        norm_each_channel=False,
        rescale=True,
        model=model,
        model_config=model_config,
        metrics=["MAE", "RMSE", "MAPE"],
        taskflow=DCRNNTaskFlow(),
        callbacks=[EarlyStopping(patience=10)],
        gpus="0,1",
        num_epochs=100,
        batch_size=64,
        lr=1e-3,
        lr_scheduler = MultiStepLR,
        lr_scheduler_params={
            "milestones": [30, 40, 50, 60, 70, 80, 90],
            "gamma": 0.1
        },
        test_interval=1000,
        eval_horizons=[3, 6, 12],
        seed=42,
        deterministic=True,
        save_results=True,
    )
    cfg.update(overrides)
    return BasicTSForecastingConfig(**cfg)
