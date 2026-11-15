from basicts import BasicTSLauncher

from src.model.tcn import TCN, TCNConfig
from src.train.config import make_config

# Task 4: which inputs help? Same TCN, same 10-epoch budget, compared on VALIDATION MAE.
VARIANTS = {
    'speed+tod':      dict(model=dict(use_tod=True)),
    'speed+tod+dow':  dict(model=dict(use_tod=True, use_dow=True)),
    'speed_in24':     dict(model=dict(input_len=24, num_layers=5), input_len=24),  # 5 layers: sees 32 >= 24 steps
    'speed_in36':     dict(model=dict(input_len=36, num_layers=6), input_len=36),  # 6 layers: sees 64 >= 36 steps
}

if __name__ == '__main__':
    for name, v in VARIANTS.items():
        BasicTSLauncher.launch_training(make_config(
            TCN, TCNConfig(**v['model']),
            input_len=v.get('input_len', 12),
            num_epochs=10,
            ckpt_save_dir=f'checkpoints/features/TCN_{name}'))
