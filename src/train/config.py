from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping

from src.train.scaler import TrainZScoreScaler
from src.train.taskflow import DCRNNTaskFlow


def make_config(model, model_config, patience=10, **overrides):
    """Shared settings for every model, so all models follow the same rules."""
    settings = dict(
        dataset_name='METR-LA',
        dataset_params={'data_file_path': 'data/METR-LA'},
        input_len=12, output_len=12,          # 1 hour in, 1 hour out
        use_timestamps=True,                  # models may take inputs_timestamps
        null_val=0.0,                         # 0 = missing
        scaler=TrainZScoreScaler,
        norm_each_channel=False,              # one mean/std for all sensors
        rescale=True,                         # loss and metrics in mph
        taskflow=DCRNNTaskFlow(),
        metrics=['MAE', 'RMSE', 'MAPE'],
        eval_horizons=[3, 6, 12],             # 15 / 30 / 60 min
        num_epochs=100,
        batch_size=64,
        lr=1e-3,
        test_interval=1000,                   # test only once, at the end
        gpus=None,                            # CPU
        seed=42, deterministic=True,
        save_results=True,
    )
    settings.update(overrides)
    return BasicTSForecastingConfig(model=model, model_config=model_config,
                                    callbacks=[EarlyStopping(patience=patience)], **settings)
